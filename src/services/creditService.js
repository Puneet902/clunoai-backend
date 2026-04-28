const supabaseAdmin = require('../config/supabaseAdmin');

const DEDUCTION_RULES = {
    audio: 1,
    screen: 2
};

const deductCredits = async (user_id, action_type) => {
    const pointsToDeduct = DEDUCTION_RULES[action_type];

    if (!pointsToDeduct) {
        throw new Error(`Invalid action type: ${action_type}`);
    }

    // 1. Get current balance
    const { data: profile, error: profileError } = await supabaseAdmin
        .from('user_profiles')
        .select('credits_remaining')
        .eq('id', user_id)
        .single();

    if (profileError) {
        throw new Error(`Error fetching balance: ${profileError.message}`);
    }

    if (profile.credits_remaining < pointsToDeduct) {
        return { success: false, message: 'Insufficient credits' };
    }

    // 2. Deduct credits
    const newBalance = profile.credits_remaining - pointsToDeduct;
    const { data: updatedProfile, error: updateError } = await supabaseAdmin
        .from('user_profiles')
        .update({ credits_remaining: newBalance })
        .eq('id', user_id)
        .select()
        .single();

    if (updateError) {
        throw new Error(`Error deducting credits: ${updateError.message}`);
    }

    // 3. Log usage
    await supabaseAdmin
        .from('usage_logs')
        .insert([{
            user_id,
            action_type,
            credits_deducted: pointsToDeduct,
            created_at: new Date()
        }]);

    return { success: true, credits_remaining: updatedProfile.credits_remaining };
};

module.exports = { deductCredits };
