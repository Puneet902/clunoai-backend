const supabaseAdmin = require('../config/supabaseAdmin');

/**
 * Initialize user if it's their first time on this device
 * Grants 10 credits if device is new.
 */
const initializeUser = async (user_id, device_hash) => {
    // 1. Check if device has already been used for a trial
    const { data: deviceRecord, error: deviceError } = await supabaseAdmin
        .from('trial_devices')
        .select('*')
        .eq('device_hash', device_hash)
        .single();

    if (deviceError && deviceError.code !== 'PGRST116') { // PGRST116 is "No rows found"
        throw new Error(`Error checking trial devices: ${deviceError.message}`);
    }

    if (deviceRecord) {
        return { success: false, message: 'Free trial already claimed on this device' };
    }

    // 2. Check user's current trial status
    const { data: profile, error: profileError } = await supabaseAdmin
        .from('user_profiles')
        .select('trial_used, credits_remaining')
        .eq('id', user_id)
        .single();

    if (profileError) {
        throw new Error(`Error fetching user profile: ${profileError.message}`);
    }

    if (profile.trial_used) {
        return { success: false, message: 'Free trial already used for this account' };
    }

    // 3. Perform atomic updates (simulated here, ideally a transaction or RPC)
    // Insert device hash
    const { error: insertError } = await supabaseAdmin
        .from('trial_devices')
        .insert([{ device_hash, user_id, used_at: new Date() }]);

    if (insertError) {
        throw new Error(`Error registering device: ${insertError.message}`);
    }

    // Update credits and mark trial as used
    const { data: updatedProfile, error: updateError } = await supabaseAdmin
        .from('user_profiles')
        .update({
            credits_remaining: profile.credits_remaining + 10,
            trial_used: true
        })
        .eq('id', user_id)
        .select()
        .single();

    if (updateError) {
        throw new Error(`Error updating user profile: ${updateError.message}`);
    }

    return { success: true, profile: updatedProfile };
};

module.exports = { initializeUser };
