const admin = require('./src/config/supabaseAdmin');

async function test() {
    const email = `test_pay_${Date.now()}@example.com`;
    const password = 'password123';

    console.log(`Creating user: ${email}`);
    const { data: { user }, error: createError } = await admin.auth.admin.createUser({
        email,
        password,
        email_confirm: true
    });

    if (createError) {
        console.error('Create Error:', createError);
        return;
    }

    console.log(`User created: ${user.id}`);

    // Create profile if not automatic
    const { error: profileError } = await admin.from('user_profiles').insert([{
        id: user.id,
        credits_remaining: 0,
        trial_used: false
    }]);

    if (profileError) {
        console.log('Profile exists or error (skipping):', profileError.message);
    }

    const { data: { session }, error: loginError } = await admin.auth.signInWithPassword({
        email,
        password
    });

    if (loginError) {
        console.error('Login Error:', loginError);
        return;
    }

    const jwt = session.access_token;
    console.log('JWT Generated');

    // Call create-order
    const response = await fetch('http://localhost:5000/api/v1/create-order', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${jwt}`
        },
        body: JSON.stringify({ plan: '599' })
    });

    const orderData = await response.json();
    console.log('Order Response:', JSON.stringify(orderData, null, 2));
}

test();
