const express = require('express');
const router = express.Router();
const { verifySupabaseJWT } = require('../middleware/authMiddleware');
const trialService = require('../services/trialService');
const creditService = require('../services/creditService');
const planService = require('../services/planService');
const resumeService = require('../services/resumeService');
const supabaseAdmin = require('../config/supabaseAdmin');
const razorpay = require('../config/razorpay');
const crypto = require('crypto');
const multer = require('multer');
const path = require('path');
const fs = require('fs');

// Configure Multer for resume uploads
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        const uploadPath = path.join(__dirname, '../../uploads');
        if (!fs.existsSync(uploadPath)) {
            fs.mkdirSync(uploadPath, { recursive: true });
        }
        cb(null, uploadPath);
    },
    filename: (req, file, cb) => {
        const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
        cb(null, file.fieldname + '-' + uniqueSuffix + path.extname(file.originalname));
    }
});

const upload = multer({
    storage: storage,
    limits: { fileSize: 5 * 1024 * 1024 }, // 5MB limit
    fileFilter: (req, file, cb) => {
        const allowedTypes = ['.pdf', '.txt', '.doc', '.docx'];
        const ext = path.extname(file.originalname).toLowerCase();
        if (allowedTypes.includes(ext)) {
            cb(null, true);
        } else {
            cb(new Error('Invalid file type. Only PDF, TXT, and DOC are allowed.'));
        }
    }
});

/**
 * POST /api/v1/payment-webhook
 * Razorpay webhook for payment verification and credit addition
 */
router.post('/payment-webhook', async (req, res, next) => {
    const secret = process.env.RAZORPAY_WEBHOOK_SECRET;
    const signature = req.headers['x-razorpay-signature'];

    console.log('🔔 Webhook received');

    try {
        const expectedSignature = crypto
            .createHmac('sha256', secret)
            .update(req.rawBody || '')
            .digest('hex');

        if (expectedSignature !== signature) {
            console.error('❌ Invalid signature');
            return res.status(400).json({ error: 'Invalid signature' });
        }

        const payload = req.body;

        // Only handle payment.captured or order.paid events
        if (payload.event === 'payment.captured' || payload.event === 'order.paid') {
            const payment_id = payload.payload.payment?.entity?.id;
            const order_id = payload.payload.payment?.entity?.order_id || payload.payload.order?.entity?.id;

            console.log(`[WEBHOOK] Received payment_id=${payment_id}`);

            const notes = payload.payload.order?.entity?.notes || 
                          payload.payload.payment?.entity?.notes;

            if (!notes || !notes.user_id || !notes.credits) {
                console.error('❌ Missing metadata in webhook payload');
                return res.status(400).json({ error: 'Missing metadata' });
            }

            const { user_id, credits } = notes;
            const creditsToAdd = parseInt(credits);

            // Check for existing payment to ensure idempotency
            const { data: existingPayment } = await supabaseAdmin
                .from('payments')
                .select('id')
                .eq('razorpay_payment_id', payment_id)
                .single();

            if (existingPayment) {
                console.log('[WEBHOOK] Duplicate payment detected. Skipping credit update.');
                return res.json({ status: 'ok', message: 'Duplicate webhook ignored' });
            }

            console.log(`💳 Processing payment for user ${user_id}: adding ${creditsToAdd} credits`);

            // Update user profile
            const { data: profile, error: fetchError } = await supabaseAdmin
                .from('user_profiles')
                .select('credits_remaining')
                .eq('id', user_id)
                .single();

            if (fetchError) throw fetchError;

            const { error: updateError } = await supabaseAdmin
                .from('user_profiles')
                .update({ credits_remaining: profile.credits_remaining + creditsToAdd })
                .eq('id', user_id);

            if (updateError) throw updateError;

            // Log payment
            await supabaseAdmin
                .from('payments')
                .insert([{
                    user_id,
                    razorpay_order_id: order_id,
                    razorpay_payment_id: payment_id,
                    amount: payload.payload.payment?.entity?.amount / 100,
                    credits_added: creditsToAdd,
                    status: 'completed',
                    created_at: new Date()
                }]);

            console.log(`✅ Success: Added ${creditsToAdd} credits to user ${user_id}`);
        }

        res.json({ status: 'ok' });
    } catch (err) {
        console.error('❌ Webhook error:', err);
        res.status(500).json({ error: 'Internal server error' });
    }
});

/**
 * GET /api/v1/plans
 * Get available subscription plans
 */
router.get('/plans', (req, res) => {
    res.json(planService.getPlans());
});

// Apply auth middleware to all routes below this point
router.use(verifySupabaseJWT);

/**
 * POST /api/v1/init-user
 * Initialize free trial based on device_hash
 */
router.post('/init-user', async (req, res, next) => {
    const { device_hash } = req.body;
    const user_id = req.user.id;

    if (!device_hash) {
        return res.status(400).json({ error: 'device_hash is required' });
    }

    try {
        const result = await trialService.initializeUser(user_id, device_hash);
        if (!result.success) {
            return res.status(400).json({ error: result.message });
        }
        res.json({ message: 'Free trial granted', profile: result.profile });
    } catch (err) {
        next(err);
    }
});

/**
 * GET /api/v1/me
 * Get current user's wallet/profile status
 */
router.get('/me', async (req, res, next) => {
    const user_id = req.user.id;

    try {
        console.log(`🔍 Fetching profile for user: ${user_id}`);
        const { data: profile, error } = await supabaseAdmin
            .from('user_profiles')
            .select('credits_remaining, trial_used')
            .eq('id', user_id)
            .single();

        if (error) {
            // PGRST116: JSON object requested, but no rows returned
            if (error.code === 'PGRST116') {
                console.log(`🆕 Creating new profile for user: ${user_id}`);
                const { data: newProfile, error: createError } = await supabaseAdmin
                    .from('user_profiles')
                    .insert([{ id: user_id, credits_remaining: 0, trial_used: false }])
                    .select()
                    .single();
                
                if (createError) {
                    console.error('❌ Profile Creation Error:', createError);
                    throw createError;
                }
                return res.json(newProfile);
            }
            console.error('❌ Profile Fetch Error:', error);
            throw error;
        }
        res.json(profile);
    } catch (err) {
        // Ensure err is never undefined when calling next()
        next(err || new Error('Unknown error in /me route'));
    }
});

/**
 * POST /api/v1/upload-resume
 * Upload a resume file
 */
router.post('/upload-resume', upload.single('resume'), async (req, res, next) => {
    const user_id = req.user.id;
    const file = req.file;

    if (!file) {
        return res.status(400).json({ error: 'No file uploaded' });
    }

    try {
        const fileUrl = `/uploads/${file.filename}`;
        const record = await resumeService.saveResumeRecord(user_id, file.originalname, fileUrl);
        res.json({ message: 'Resume uploaded successfully', resume: record });
    } catch (err) {
        // Cleanup uploaded file if DB insert fails
        if (file) {
            try { fs.unlinkSync(file.path); } catch (e) {}
        }
        next(err);
    }
});

/**
 * GET /api/v1/resumes
 * Get user's uploaded resumes
 */
router.get('/resumes', async (req, res, next) => {
    const user_id = req.user.id;
    try {
        const resumes = await resumeService.getUserResumes(user_id);
        res.json(resumes);
    } catch (err) {
        next(err);
    }
});

/**
 * POST /api/v1/create-order
 * Create a new Razorpay order
 */
router.post('/create-order', async (req, res, next) => {
    const { planId } = req.body;
    const user_id = req.user.id;

    const selectedPlan = planService.getPlanById(planId);

    if (!selectedPlan) {
        return res.status(400).json({ error: 'Invalid plan selected' });
    }

    try {
        const options = {
            amount: selectedPlan.amount * 100, // Razorpay expects amount in paise
            currency: 'INR',
            receipt: `rcpt_${Math.floor(Math.random() * 1000000)}`,
            notes: {
                user_id,
                credits: selectedPlan.credits
            }
        };

        const order = await razorpay.orders.create(options);
        console.log(`✅ Order created: ${order.id} for user ${user_id}`);

        res.json({
            order_id: order.id,
            amount: order.amount,
            key_id: process.env.RAZORPAY_KEY_ID,
            credits: selectedPlan.credits
        });
    } catch (err) {
        console.error('❌ Razorpay Order Error:', err);
        next(err);
    }
});

/**
 * POST /api/v1/verify-payment
 * Verify a payment from the frontend and add credits
 */
router.post('/verify-payment', async (req, res, next) => {
    const { razorpay_order_id, razorpay_payment_id, razorpay_signature, planId } = req.body;
    const user_id = req.user.id;

    try {
        // 1. Verify Signature
        const secret = process.env.RAZORPAY_KEY_SECRET;
        const generated_signature = crypto
            .createHmac('sha256', secret)
            .update(razorpay_order_id + "|" + razorpay_payment_id)
            .digest('hex');

        if (generated_signature !== razorpay_signature) {
            return res.status(400).json({ error: 'Invalid payment signature' });
        }

        // 2. Add Credits
        const selectedPlan = planService.getPlanById(planId);
        if (!selectedPlan) {
            return res.status(400).json({ error: 'Invalid plan' });
        }

        console.log(`💳 Manually verifying payment for user ${user_id}: adding ${selectedPlan.credits} credits`);

        // Check for existing payment to prevent duplicates
        const { data: existingPayment } = await supabaseAdmin
            .from('payments')
            .select('id')
            .eq('razorpay_payment_id', razorpay_payment_id)
            .single();

        if (existingPayment) {
            return res.json({ status: 'ok', message: 'Credits already added' });
        }

        // Update user profile (Create if doesn't exist)
        const { data: profile, error: profileError } = await supabaseAdmin
            .from('user_profiles')
            .select('credits_remaining')
            .eq('id', user_id)
            .single();

        if (profileError && profileError.code !== 'PGRST116') {
            console.error('❌ Supabase Error (Profile Fetch):', profileError);
            throw new Error(`Profile fetch failed: ${profileError.message}`);
        }

        const currentCredits = profile?.credits_remaining || 0;
        console.log(`🏦 Current credits for ${user_id}: ${currentCredits}`);

        const { error: upsertError } = await supabaseAdmin
            .from('user_profiles')
            .upsert({ 
                id: user_id, 
                credits_remaining: currentCredits + selectedPlan.credits
            }, { onConflict: 'id' });

        if (upsertError) {
            console.error('❌ Supabase Error (Profile Upsert):', upsertError);
            throw new Error(`Profile upsert failed: ${upsertError.message}`);
        }

        console.log(`✅ Upserted ${selectedPlan.credits} credits. New total: ${currentCredits + selectedPlan.credits}`);

        // Log payment
        await supabaseAdmin
            .from('payments')
            .insert([{
                user_id,
                razorpay_order_id,
                razorpay_payment_id,
                amount: selectedPlan.amount,
                credits_added: selectedPlan.credits,
                status: 'completed',
                created_at: new Date()
            }]);

        res.json({ status: 'ok', message: 'Payment verified and credits added' });
    } catch (err) {
        console.error('❌ Verification Error:', err);
        next(err);
    }
});

/**
 * POST /api/v1/deduct
 * Deduct credits for an action
 */
router.post('/deduct', async (req, res, next) => {
    const { action_type } = req.body;
    const user_id = req.user.id;

    if (!action_type) {
        return res.status(400).json({ error: 'action_type is required' });
    }

    try {
        const result = await creditService.deductCredits(user_id, action_type);
        if (!result.success) {
            return res.status(402).json({ error: result.message });
        }
        res.json({ message: 'Credits deducted', credits_remaining: result.credits_remaining });
    } catch (err) {
        next(err);
    }
});

/**
 * GET /api/v1/usage
 * Get user's payment and usage history
 */
router.get('/usage', async (req, res, next) => {
    const user_id = req.user.id;
    try {
        const { data: payments, error } = await supabaseAdmin
            .from('payments')
            .select('*')
            .eq('user_id', user_id)
            .order('created_at', { ascending: false });

        if (error) throw error;

        // Map payments to the format expected by the dashboard
        const activities = payments.map(p => ({
            action: `Added ${p.credits_added} Credits`,
            credits_used: -p.credits_added, // negative used = positive added in UI
            created_at: p.created_at
        }));

        res.json(activities);
    } catch (err) {
        next(err);
    }
});

module.exports = router;
