const supabaseAdmin = require('../config/supabaseAdmin');
const creditService = require('./creditService');
const planService = require('./planService');

/**
 * BridgeService listens to Supabase Realtime 'webhook_events' table.
 * This allows a public Supabase Edge Function to receive Razorpay webhooks
 * and "broadcast" them to our local backend without needing a tunnel.
 */
class BridgeService {
    constructor() {
        this.subscription = null;
    }

    start() {
        console.log('🔗 Supabase Webhook Bridge starting...');

        this.subscription = supabaseAdmin
            .channel('webhook_bridge')
            .on(
                'postgres_changes',
                { event: 'INSERT', schema: 'public', table: 'webhook_events' },
                (payload) => {
                    this.handleWebhookEvent(payload.new);
                }
            )
            .subscribe((status) => {
                console.log(`📡 Realtime Subscription Status: ${status}`);
            });
    }

    async handleWebhookEvent(event) {
        console.log('🔔 Bridge: Received webhook from Supabase');
        const { payload } = event;

        try {
            // We only care about success events
            if (payload.event === 'payment.captured' || payload.event === 'order.paid') {
                const notes = payload.payload.order?.entity?.notes || 
                              payload.payload.payment?.entity?.notes;

                if (!notes || !notes.user_id || !notes.credits) {
                    console.error('❌ Bridge: Missing metadata in payload');
                    return;
                }

                const { user_id, credits } = notes;
                const creditsToAdd = parseInt(credits);

                console.log(`💳 Bridge: Processing ${creditsToAdd} credits for user ${user_id}`);

                // Update Profile Logic (Reusing code from creditService or manual update)
                const { data: profile } = await supabaseAdmin
                    .from('user_profiles')
                    .select('credits_remaining')
                    .eq('id', user_id)
                    .single();

                const currentCredits = profile?.credits_remaining || 0;

                const { error: updateError } = await supabaseAdmin
                    .from('user_profiles')
                    .upsert({ 
                        id: user_id, 
                        credits_remaining: currentCredits + creditsToAdd
                    }, { onConflict: 'id' });

                if (updateError) throw updateError;

                console.log(`✅ Bridge Success: Added ${creditsToAdd} credits to ${user_id}`);
                
                // Optional: Delete the event from Supabase after processing to keep it clean
                await supabaseAdmin.from('webhook_events').delete().eq('id', event.id);
            }
        } catch (err) {
            console.error('❌ Bridge Error:', err);
        }
    }

    stop() {
        if (this.subscription) {
            this.subscription.unsubscribe();
        }
    }
}

module.exports = new BridgeService();
