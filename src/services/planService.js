const plans = [
    {
        id: 'plan_1',
        name: 'Basic',
        credits: 100,
        amount: 499, // in INR
        display_price: '₹499'
    },
    {
        id: 'plan_2',
        name: 'Professional',
        credits: 250,
        amount: 999, // in INR
        display_price: '₹999'
    }
];

const getPlans = () => {
    return plans;
};

const getPlanById = (planId) => {
    return plans.find(p => p.id === planId) || plans.find(p => p.amount === parseInt(planId));
};

module.exports = {
    getPlans,
    getPlanById
};
