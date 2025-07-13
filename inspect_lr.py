import matplotlib.pyplot as plt
import torch

num_steps = 20
initial_lr = 3e-4

optimizer = torch.optim.Adam([torch.zeros(1)], lr=initial_lr)

# StepLR
step_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.794)
step_lrs = []
for _ in range(num_steps):
    step_scheduler.step()
    step_lrs.append(optimizer.param_groups[0]['lr'])

optimizer = torch.optim.Adam([torch.zeros(1)], lr=initial_lr)

# ExponentialLR
exp_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
exp_lrs = []
for _ in range(num_steps):
    exp_scheduler.step()
    exp_lrs.append(optimizer.param_groups[0]['lr'])

optimizer = torch.optim.Adam([torch.zeros(1)], lr=initial_lr)

# CosineAnnealingLR
cos_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=1e-4)
cos_lrs = []
for _ in range(num_steps):
    cos_scheduler.step()
    cos_lrs.append(optimizer.param_groups[0]['lr'])

optimizer = torch.optim.Adam([torch.zeros(1)], lr=initial_lr)

# # CyclicLR
# cyclic_scheduler = torch.optim.lr_scheduler.CyclicLR(
#     optimizer, base_lr=1e-5, max_lr=initial_lr, step_size_up=num_steps // 2, mode='triangular'
# )
# cyclic_lrs = []
# for _ in range(num_steps):
#     cyclic_scheduler.step()
#     cyclic_lrs.append(optimizer.param_groups[0]['lr'])

# Plot
plt.figure(figsize=(10, 6))
plt.plot(step_lrs, label='StepLR')
# plt.plot(exp_lrs, label='ExponentialLR (gamma=0.99)')
plt.plot(cos_lrs, label='CosineAnnealingLR ')
# plt.plot(cyclic_lrs, label='CyclicLR (triangular)')

plt.xlabel('Step')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedulers Comparison (Start LR=1e-3)')
plt.legend()
plt.grid(True)
plt.savefig('lr_schedulers_comparison.png')