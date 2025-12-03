"""
Train RL policy for query mutation selection
Simplified version of RLbreaker's train_policy.py
"""

import os
import sys
import signal
import torch
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import csv
from datetime import datetime

# Import RL core components
from rl_core import PPO, Policy, RolloutStorage
from rl_query_mutator_env import QueryMutationEnv, BatchedQueryMutationEnv


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
    parser.add_argument('--target-model', type=str, default='llava:latest',
                        help='target model to attack')
    parser.add_argument('--mutator-model', type=str, default='gemma3:latest',
                        help='model for generating mutations')
    parser.add_argument('--judge-model', type=str, default='deepseek-r1:14b',
                        help='model for judging responses')
    parser.add_argument('--uncensored-model', type=str, default='wizard-vicuna-uncensored',
                        help='uncensored model for generating unaligned responses')
    
    # Reward mechanism
    parser.add_argument('--use-llm-judge', action='store_true',
                        help='use LLM judge for reward (slower but more accurate)')
    
    # Pregenerated responses
    parser.add_argument('--unaligned-csv', type=str, default='dataset/prompts_harmful_responses.csv',
                        help='CSV file with pregenerated unaligned responses')
    parser.add_argument('--use-unified-csv', action='store_true', default=True,
                        help='load both queries and responses from the same CSV file (unaligned-csv)')
    
    # Dataset sampling
    parser.add_argument('--frac-samples', type=float, default=1.0,
                        help='fraction of dataset to randomly sample (0.0-1.0, default: 1.0 = all data)')
    
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
    
    # Batching
    parser.add_argument('--use-batching', action='store_true', default=True,
                        help='use batched operations for faster training (default: True)')
    parser.add_argument('--no-batching', action='store_false', dest='use_batching',
                        help='disable batched operations')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='number of concurrent API calls in batched operations (default: 8)')
    
    # Image-based prompts
    parser.add_argument('--no-image-prompts', action='store_true',
                        help='disable image-based prompts and use text prompts instead')
    parser.add_argument('--image-style', type=str, default='simple_text',
                        choices=['simple_text', 'stepwise', 'archaic_english', 'technical_jargon', 'highlighted', 'multi_line'],
                        help='style for image-based prompts (default: simple_text)')
    parser.add_argument('--save-images', action='store_true',
                        help='save generated images for debugging')
    
    args = parser.parse_args()
    
    # Image prompts are enabled by default
    args.use_image_prompts = not args.no_image_prompts
    
    # Set device
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")
    
    # Set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    # Create timestamped run directory under logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join('logs', f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    
    # Update save_dir to use the timestamped run directory
    args.save_dir = run_dir
    
    # Save run configuration to separate log file
    config_path = os.path.join(run_dir, 'run.log')
    with open(config_path, 'w', encoding='utf-8') as config_file:
        config_file.write("=" * 60 + "\n")
        config_file.write("Run Configuration\n")
        config_file.write("=" * 60 + "\n")
        config_file.write(f"Timestamp: {timestamp}\n")
        config_file.write(f"Run Directory: {run_dir}\n")
        config_file.write("\n")
        config_file.write("Models:\n")
        config_file.write(f"  Target Model: {args.target_model}\n")
        config_file.write(f"  Mutator Model: {args.mutator_model}\n")
        config_file.write(f"  Judge Model: {args.judge_model}\n")
        config_file.write(f"  Uncensored Model: {args.uncensored_model}\n")
        config_file.write("\n")
        config_file.write("Training Parameters:\n")
        config_file.write(f"  Num Env Steps: {args.num_env_steps}\n")
        config_file.write(f"  Num Steps: {args.num_steps}\n")
        config_file.write(f"  Num Processes: {args.num_processes}\n")
        config_file.write(f"  Learning Rate: {args.lr}\n")
        config_file.write(f"  Gamma: {args.gamma}\n")
        config_file.write(f"  PPO Epochs: {args.ppo_epoch}\n")
        config_file.write(f"  Num Mini Batch: {args.num_mini_batch}\n")
        config_file.write(f"  Clip Param: {args.clip_param}\n")
        config_file.write(f"  Value Loss Coef: {args.value_loss_coef}\n")
        config_file.write(f"  Entropy Coef: {args.entropy_coef}\n")
        config_file.write(f"  Max Grad Norm: {args.max_grad_norm}\n")
        config_file.write("\n")
        config_file.write("Reward & Dataset:\n")
        config_file.write(f"  Use LLM Judge: {args.use_llm_judge}\n")
        config_file.write(f"  Unaligned CSV: {args.unaligned_csv}\n")
        config_file.write(f"  Use Unified CSV: {args.use_unified_csv}\n")
        config_file.write(f"  Fraction Samples: {args.frac_samples}\n")
        config_file.write("\n")
        config_file.write("Batching:\n")
        config_file.write(f"  Use Batching: {args.use_batching}\n")
        config_file.write(f"  Batch Size: {args.batch_size}\n")
        config_file.write("\n")
        config_file.write("Image-Based Prompts:\n")
        config_file.write(f"  Use Image Prompts: {args.use_image_prompts}\n")
        config_file.write(f"  Image Style: {args.image_style}\n")
        config_file.write(f"  Save Images: {args.save_images}\n")
        config_file.write("\n")
        config_file.write("Other:\n")
        config_file.write(f"  Device: {'cuda' if args.cuda else 'cpu'}\n")
        config_file.write(f"  Seed: {args.seed}\n")
        config_file.write(f"  Save Interval: {args.save_interval}\n")
        config_file.write(f"  Log Interval: {args.log_interval}\n")
        config_file.write("=" * 60 + "\n")
    
    # Create CSV log file for training events
    csv_path = os.path.join(run_dir, 'training_log.csv')
    csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    
    # Write column headers only
    csv_writer.writerow(['step', 'update', 'episode', 'query', 'mutation_type', 'mutated_query', 
                         'target_response', 'unaligned_response', 'reward_score', 'mutation_number'])
    
    print("="*60)
    print("RL Query Mutation Training")
    print("="*60)
    print(f"Run Directory: {run_dir}")
    print(f"Target Model: {args.target_model}")
    print(f"Mutator Model: {args.mutator_model}")
    print(f"Judge Model: {args.judge_model}")
    print(f"Uncensored Model: {args.uncensored_model}")
    print(f"Use LLM Judge: {args.use_llm_judge}")
    print(f"Use Batching: {args.use_batching}")
    if args.use_batching:
        print(f"Batch Size: {args.batch_size} concurrent API calls")
    print(f"Use Image Prompts: {args.use_image_prompts}")
    if args.use_image_prompts:
        print(f"Image Style: {args.image_style}")
    print(f"Save Images: {args.save_images}")
    print(f"Device: {device}")
    print("="*60)
    
    # Get image style enum if using image prompts
    image_style = None
    if args.use_image_prompts:
        from image_prompt_generator import ImagePromptStyle
        image_style = ImagePromptStyle[args.image_style]
    
    # Observation size (nomic-embed-text has 768 dimensions)
    obs_size = 768
    
    # Create dummy environment to get spaces
    dummy_env = QueryMutationEnv(args, obs_size, eval=False, use_image_prompts=args.use_image_prompts, image_style=image_style)
    
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
        env = QueryMutationEnv(args, obs_size, eval=False, use_image_prompts=args.use_image_prompts, image_style=image_style)
        envs.append(env)
    
    # Wrap environments in batched wrapper if batching is enabled
    if args.use_batching:
        batched_env = BatchedQueryMutationEnv(envs, batch_size=args.batch_size)
        print("Batched operations enabled for faster training")
    else:
        batched_env = None
        print("Batched operations disabled")
    
    # Initialize environments
    if args.use_batching:
        obs = batched_env.batch_reset()
    else:
        obs = []
        for env in envs:
            obs.append(env.reset())
        obs = np.array(obs)
    
    obs = torch.FloatTensor(obs).to(device)
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
    
    try:
        for update in range(num_updates):
            # Collect rollouts
            for step in range(args.num_steps):
                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step]
                    )
                
                # Execute actions in all environments
                action_indices = action.cpu().numpy()
                from rl_query_mutator_env import QueryMutator
                
                if args.use_batching:
                    # Use batched operations for faster processing
                    obs_batch, reward_batch, done_batch, info_batch = batched_env.batch_step(action_indices)
                else:
                    # Sequential processing (original behavior)
                    obs_batch = []
                    reward_batch = []
                    done_batch = []
                    info_batch = []
                    
                    for i, env in enumerate(envs):
                        action_idx = action_indices[i]
                        obs, reward, done, info = env.step(action_idx)
                        
                        if done:
                            obs = env.reset()
                        
                        obs_batch.append(obs)
                        reward_batch.append(reward)
                        done_batch.append(done)
                        info_batch.append(info)
                    
                    obs_batch = np.array(obs_batch)
                    reward_batch = np.array(reward_batch)
                
                # Log to CSV and handle episode resets
                for i, (obs_i, reward_i, done_i, info_i) in enumerate(zip(obs_batch, reward_batch, done_batch, info_batch)):
                    action_idx = action_indices[i]
                    mutation_name = QueryMutator(action_idx).name
                    csv_writer.writerow([
                        total_steps + i,
                        update,
                        episode_count,
                        info_i.get('original_query', '')[:100],  # Truncate for readability
                        mutation_name,
                        info_i.get('mutated_query', '')[:100],
                        info_i.get('target_response', '')[:100],
                        info_i.get('unaligned_response', '')[:100] if 'unaligned_response' in info_i else '',
                        reward_i,
                        envs[i].steps
                    ])
                    
                    if done_i:
                        episode_rewards.append(info_i.get('reward', 0))
                        episode_lengths.append(envs[i].steps)
                        episode_count += 1
                        # Reset the environment if using batching (already done in sequential mode)
                        if args.use_batching:
                            obs_batch[i] = envs[i].reset()
                
                # Convert to tensors
                obs = torch.FloatTensor(obs_batch).to(device)
                rewards = torch.FloatTensor(reward_batch).unsqueeze(1).to(device)
                masks = torch.FloatTensor([[0.0] if done else [1.0] for done in done_batch]).to(device)
                
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
    
    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print("Training interrupted by user (Ctrl+C)")
        print(f"Completed {total_steps} steps before interruption")
        print("="*60)
    
    finally:
        # Close progress bar
        if pbar:
            pbar.close()
        
        # Close CSV file
        if csv_file and not csv_file.closed:
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
        print(f"Run Directory: {run_dir}")
        print(f"Config saved to {config_path}")
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
