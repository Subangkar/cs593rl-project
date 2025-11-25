"""
Train RL policy for query mutation selection
Simplified version of RLbreaker's train_policy.py
"""

import os
import torch
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import csv
from datetime import datetime

# Import standalone PPO implementation
from ppo_algorithm import PPO
from policy_network import Policy
from rollout_storage import RolloutStorage
from rl_query_mutator_env import QueryMutationEnv


def main():
    parser = argparse.ArgumentParser(description='Train RL policy for query mutations')
    
    # Training parameters
    parser.add_argument('--num-steps', type=int, default=32,
                        help='number of forward steps in PPO (default: 32)')
    parser.add_argument('--num-processes', type=int, default=1,
                        help='number of parallel environments (default: 1)')
    parser.add_argument('--num-env-steps', type=int, default=10000,
                        help='total environment steps (default: 10000)')
    
    # PPO parameters
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='learning rate (default: 3e-4)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor (default: 0.99)')
    parser.add_argument('--clip-param', type=float, default=0.2,
                        help='PPO clip parameter (default: 0.2)')
    parser.add_argument('--ppo-epoch', type=int, default=4,
                        help='number of PPO epochs (default: 4)')
    parser.add_argument('--num-mini-batch', type=int, default=4,
                        help='number of mini-batches (default: 4)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='entropy coefficient (default: 0.01)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    
    # Model parameters
    parser.add_argument('--target-model', type=str, default='llama3.1:8b',
                        help='target model to attack')
    parser.add_argument('--mutator-model', type=str, default='deepseek-r1:14b',
                        help='model for generating mutations')
    parser.add_argument('--judge-model', type=str, default='deepseek-r1:14b',
                        help='model for judging responses')
    parser.add_argument('--uncensored-model', type=str, default='wizard-vicuna-uncensored',
                        help='uncensored model for generating unaligned responses')
    
    # Reward mechanism
    parser.add_argument('--use-llm-judge', action='store_true',
                        help='use LLM judge for reward (slower but more accurate)')
    
    # Checkpoint
    parser.add_argument('--save-dir', type=str, default='./trained_models_query_mutator',
                        help='directory to save checkpoints')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='save model every N updates')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='log info every N updates')
    
    # Device
    parser.add_argument('--no-cuda', action='store_true',
                        help='disable CUDA training')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    
    args = parser.parse_args()
    
    # Set device
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")
    
    # Set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Create CSV log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.save_dir, f'training_log_{timestamp}.csv')
    csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['step', 'update', 'episode', 'query', 'mutation_type', 'mutated_query', 
                         'target_response', 'unaligned_response', 'reward_score', 'mutation_number'])
    
    print("="*60)
    print("RL Query Mutation Training")
    print("="*60)
    print(f"Target Model: {args.target_model}")
    print(f"Mutator Model: {args.mutator_model}")
    print(f"Judge Model: {args.judge_model}")
    print(f"Uncensored Model: {args.uncensored_model}")
    print(f"Use LLM Judge: {args.use_llm_judge}")
    print(f"Device: {device}")
    print(f"Save Directory: {args.save_dir}")
    print("="*60)
    
    # Observation size (nomic-embed-text has 768 dimensions)
    obs_size = 768
    
    # Create dummy environment to get spaces
    dummy_env = QueryMutationEnv(args, obs_size, eval=False)
    
    # Create actor-critic policy
    actor_critic = Policy(
        dummy_env.observation_space.shape,
        dummy_env.action_space,
        base_kwargs={'recurrent': False, 'device': device, 'hidden_size': 64}
    )
    actor_critic.to(device)
    
    print(f"Policy Network:")
    print(f"  Observation Space: {dummy_env.observation_space.shape}")
    print(f"  Action Space: {dummy_env.action_space.n} mutations")
    print(f"  Total Parameters: {sum(p.numel() for p in actor_critic.parameters())}")
    print()
    
    # Create PPO agent
    agent = PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=1e-5,
        max_grad_norm=args.max_grad_norm
    )
    
    # Create rollout storage
    rollouts = RolloutStorage(
        args.num_steps,
        args.num_processes,
        dummy_env.observation_space.shape,
        dummy_env.action_space,
        actor_critic.recurrent_hidden_state_size
    )
    
    # Create environments
    print(f"Creating {args.num_processes} parallel environments...")
    envs = []
    for i in range(args.num_processes):
        env = QueryMutationEnv(args, obs_size, eval=False)
        envs.append(env)
    
    # Initialize environments
    obs = []
    for env in envs:
        obs.append(env.reset())
    obs = torch.FloatTensor(np.array(obs)).to(device)
    
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)
    
    # Training loop
    num_updates = args.num_env_steps // args.num_steps // args.num_processes
    
    # Ensure at least 1 update
    if num_updates == 0:
        print(f"\nWARNING: num_env_steps ({args.num_env_steps}) is too small!")
        print(f"With num_steps={args.num_steps} and num_processes={args.num_processes},")
        print(f"you need at least {args.num_steps * args.num_processes} steps for 1 update.")
        print(f"Adjusting to minimum required steps...")
        args.num_env_steps = args.num_steps * args.num_processes
        num_updates = 1
    
    print(f"\nStarting training for {num_updates} updates...")
    print(f"Total environment steps: {args.num_env_steps}")
    print()
    
    episode_rewards = []
    episode_lengths = []
    total_steps = 0
    episode_count = 0
    
    # Create progress bar
    pbar = tqdm(total=args.num_env_steps, desc="Training", unit="steps")
    
    for update in range(num_updates):
        # Collect rollouts
        for step in range(args.num_steps):
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step]
                )
            
            # Execute actions in all environments
            obs_list = []
            reward_list = []
            done_list = []
            info_list = []
            
            for i, env in enumerate(envs):
                action_idx = action[i].item()
                obs, reward, done, info = env.step(action_idx)
                
                # Log to CSV
                from rl_query_mutator_env import QueryMutator
                mutation_name = QueryMutator(action_idx).name
                csv_writer.writerow([
                    total_steps + i,
                    update,
                    episode_count,
                    info.get('original_query', '')[:100],  # Truncate for readability
                    mutation_name,
                    info.get('mutated_query', '')[:100],
                    info.get('target_response', '')[:100],
                    info.get('unaligned_response', '')[:100] if 'unaligned_response' in info else '',
                    reward,
                    env.steps
                ])
                
                if done:
                    obs = env.reset()
                    episode_rewards.append(info.get('reward', 0))
                    episode_lengths.append(env.steps)
                    episode_count += 1
                
                obs_list.append(obs)
                reward_list.append(reward)
                done_list.append(done)
                info_list.append(info)
            
            # Convert to tensors
            obs = torch.FloatTensor(np.array(obs_list)).to(device)
            rewards = torch.FloatTensor(reward_list).unsqueeze(1).to(device)
            masks = torch.FloatTensor([[0.0] if done else [1.0] for done in done_list]).to(device)
            
            # Store in rollout buffer
            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, rewards, masks)
            
            total_steps += args.num_processes
            pbar.update(args.num_processes)
        
        # Compute returns
        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]
            ).detach()
        
        rollouts.compute_returns(next_value, args.gamma, 0.95, use_gae=True)
        
        # Update policy
        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        
        rollouts.after_update()
        
        # Logging
        if update % args.log_interval == 0:
            avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0.0
            avg_length = np.mean(episode_lengths[-100:]) if episode_lengths else 0.0
            success_rate = envs[0].successful_attacks / max(1, envs[0].total_queries)
            
            # Update progress bar with metrics
            pbar.set_postfix({
                'update': f'{update}/{num_updates}',
                'reward': f'{avg_reward:.3f}',
                'success': f'{success_rate:.1%}',
                'v_loss': f'{value_loss:.3f}',
                'a_loss': f'{action_loss:.3f}'
            })
            
            # Detailed logging
            tqdm.write(f"\nUpdate {update}/{num_updates} | Steps {total_steps}/{args.num_env_steps}")
            tqdm.write(f"  Avg Reward: {avg_reward:.4f}")
            tqdm.write(f"  Avg Episode Length: {avg_length:.2f}")
            tqdm.write(f"  Success Rate: {success_rate:.2%}")
            tqdm.write(f"  Value Loss: {value_loss:.4f}")
            tqdm.write(f"  Action Loss: {action_loss:.4f}")
            tqdm.write(f"  Entropy: {dist_entropy:.4f}")
        
        # Save checkpoint
        if update % args.save_interval == 0 and update > 0:
            save_path = os.path.join(args.save_dir, f'checkpoint_{update}.pt')
            torch.save({
                'update': update,
                'model_state_dict': actor_critic.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'avg_reward': np.mean(episode_rewards[-100:]) if episode_rewards else 0.0,
            }, save_path)
            tqdm.write(f"Saved checkpoint to {save_path}")
    
    # Close progress bar
    pbar.close()
    
    # Close CSV file
    csv_file.close()
    
    # Save final model
    final_path = os.path.join(args.save_dir, 'final_model.pt')
    torch.save({
        'model_state_dict': actor_critic.state_dict(),
        'avg_reward': np.mean(episode_rewards[-100:]) if episode_rewards else 0.0,
        'total_steps': total_steps,
    }, final_path)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Final model saved to {final_path}")
    print(f"Training log saved to {csv_path}")
    print(f"Total Steps: {total_steps}")
    print(f"Final Success Rate: {envs[0].successful_attacks / max(1, envs[0].total_queries):.2%}")
    print("="*60)


if __name__ == '__main__':
    try:
        import multiprocessing
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    main()
