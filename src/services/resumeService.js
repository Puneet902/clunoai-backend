const supabaseAdmin = require('../config/supabaseAdmin');

const saveResumeRecord = async (user_id, filename, fileUrl) => {
    const { data, error } = await supabaseAdmin
        .from('resumes')
        .insert([
            {
                user_id,
                filename,
                file_url: fileUrl,
                status: 'uploaded',
                created_at: new Date()
            }
        ])
        .select()
        .single();

    if (error) throw error;
    return data;
};

const getUserResumes = async (user_id) => {
    const { data, error } = await supabaseAdmin
        .from('resumes')
        .select('*')
        .eq('user_id', user_id)
        .order('created_at', { ascending: false });

    if (error) throw error;
    return data;
};

module.exports = {
    saveResumeRecord,
    getUserResumes
};
