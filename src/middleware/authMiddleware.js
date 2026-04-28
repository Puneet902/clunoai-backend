const supabaseAdmin = require('../config/supabaseAdmin');

/**
 * verifySupabaseJWT - Middleware to verify Supabase JWT
 * Expects: Authorization: Bearer <token>
 */
const verifySupabaseJWT = async (req, res, next) => {
    // Developer bypass for local dashboard testing
    if (req.headers['x-cluno-dev'] === 'true') {
        req.user = { id: 'c54507de-6471-4a67-a9e3-2bf98fdb6f12' };
        return next();
    }

    const authHeader = req.headers.authorization;

    if (!authHeader || !authHeader.startsWith('Bearer ')) {
        return res.status(401).json({ error: 'Missing or invalid authorization header' });
    }

    const token = authHeader.split(' ')[1];

    try {
        // Verify the token with Supabase Auth
        const { data: { user }, error } = await supabaseAdmin.auth.getUser(token);

        if (error || !user) {
            return res.status(401).json({ error: 'Invalid or expired token' });
        }

        // Attach user to request object
        req.user = user;
        next();
    } catch (err) {
        console.error('Auth Middleware Error:', err);
        return res.status(500).json({ error: 'Internal server error during authentication' });
    }
};

module.exports = { verifySupabaseJWT };
