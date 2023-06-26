% check validity of custom layer
%checkLayer(CustomPositiveLayer('post1'), [32 32]);
x = checkLayer(CustomConstantAddLayer('add1', 0.2), [32 32]);
