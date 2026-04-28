const admin = require('./src/config/supabaseAdmin');

async function debugAuth() {
    console.log('--- Debugging Supabase Auth ---');

    // Try normal signup
    console.log('Attempting public signUp...');
    const { data, error } = await admin.auth.signUp({
        email: `test_${Date.now()}@example.com`,
        password: 'password123'
    });

    if (error) {
        console.log('Public signUp Error:', error.status, error.code, error.message);
    } else {
        console.log('Public signUp SUCCESS:', data.user.id);
    }

    // Try admin create
    console.log('\nAttempting admin createUser...');
    const { data: adminData, error: adminError } = await admin.auth.admin.createUser({
        email: `admin_${Date.now()}@example.com`,
        password: 'password123',
        email_confirm: true
    });

    if (adminError) {
        console.log('Admin createUser Error:', adminError.status, adminError.code, adminError.message);
    } else {
        console.log('Admin createUser SUCCESS:', adminData.user.id);
    }
}

debugAuth();
