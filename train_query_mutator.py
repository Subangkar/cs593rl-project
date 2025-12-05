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
import json
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

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
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--gamma', type=float, default=0.95,
                        help='discount factor (default: 0.95)')
    parser.add_argument('--clip-param', type=float, default=0.2,
                        help='PPO clip parameter (default: 0.2)')
    parser.add_argument('--ppo-epoch', type=int, default=4,
                        help='number of PPO epochs (default: 4)')
    parser.add_argument('--num-mini-batch', type=int, default=4,
                        help='number of mini-batches (default: 4)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--entropy-coef', type=float, default=0.03,
                        help='entropy coefficient (default: 0.03)')
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
    parser.add_argument('--unaligned-csv', type=str, default='dataset/prompts_harmful_responses_original_backup.csv',
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
    parser.add_argument('--force-gpu', action='store_true',
                        help='force GPU usage (fail if CUDA not available)')
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
    parser.add_argument('--image-style', type=str, default='stepwise',
                        choices=['simple_text', 'stepwise', 'archaic_english', 'technical_jargon', 'highlighted', 'multi_line'],
                        help='style for image-based prompts (default: simple_text)')
    parser.add_argument('--save-images', action='store_true',
                        help='save generated images for debugging')
    
    args = parser.parse_args()
    
    # Image prompts are enabled by default
    args.use_image_prompts = not args.no_image_prompts
    
    # Set device
    if args.force_gpu and not torch.cuda.is_available():
        raise RuntimeError("--force-gpu specified but CUDA is not available")
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.force_gpu:
        args.cuda = True
        
    device = torch.device("cuda:0" if args.cuda else "cpu")
    
    # Print device selection debug info
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"no_cuda flag: {args.no_cuda}")
    print(f"force_gpu flag: {args.force_gpu}")
    print(f"Using CUDA: {args.cuda}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Selected device: {device}")
    print("-" * 40)
    
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
    
    # Create TensorBoard writer
    tensorboard_dir = os.path.join(run_dir, 'tensorboard')
    writer = SummaryWriter(log_dir=tensorboard_dir)
    print(f"TensorBoard logs: {tensorboard_dir}")
    print(f"Run: tensorboard --logdir {tensorboard_dir} --port 6006")
    print("-" * 60)
    
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
    
    # Create JSON log file for training events
    json_path = os.path.join(run_dir, 'training_log.json')
    json_file = open(json_path, 'w', encoding='utf-8')
    json_file.write('[\n')  # Start JSON array
    first_entry = True  # Track if we need to add comma before entry
    
    # Create ASR log file for tracking attack success rates
    asr_log_path = os.path.join(run_dir, 'asr_logs.json')
    asr_log_file = open(asr_log_path, 'w', encoding='utf-8')
    asr_log_file.write('[\n')  # Start JSON array
    first_asr_entry = True
    
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
    
    # Create image log directory if saving images
    image_log_dir = os.path.join(run_dir, 'images') if args.save_images else None
    
    # Create dummy environment to get spaces
    dummy_env = QueryMutationEnv(args, obs_size, eval=False, use_image_prompts=args.use_image_prompts, image_style=image_style, image_log_dir=image_log_dir)
    
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
        env = QueryMutationEnv(args, obs_size, eval=False, use_image_prompts=args.use_image_prompts, image_style=image_style, image_log_dir=image_log_dir)
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
                
                # Log to JSON and handle episode resets
                for i, (obs_i, reward_i, done_i, info_i) in enumerate(zip(obs_batch, reward_batch, done_batch, info_batch)):
                    action_idx = action_indices[i]
                    mutation_name = QueryMutator(action_idx).name
                    
                    log_entry = {
                        'step': int(total_steps + i),
                        'update': int(update),
                        'episode': int(episode_count),
                        'query': info_i.get('original_query', ''),
                        'mutation_type': mutation_name,
                        'mutated_query': info_i.get('mutated_query', ''),
                        'target_response': info_i.get('target_response', ''),
                        'unaligned_response': info_i.get('unaligned_response', '') if 'unaligned_response' in info_i else '',
                        'reward_score': float(reward_i),
                        'mutation_number': int(envs[i].steps),
                        'judge_explanation': info_i.get('judge_explanation', '')
                    }
                    
                    # Write entry to JSON file
                    if not first_entry:
                        json_file.write(',\n')
                    json.dump(log_entry, json_file, indent=2)
                    json_file.flush()
                    first_entry = False
                    
                    if done_i:
                        episode_rewards.append(info_i.get('reward', 0))
                        episode_lengths.append(envs[i].steps)
                        episode_count += 1
                        # Reset the environment if using batching (already done in sequential mode)
                        if args.use_batching:
                            obs_batch[i] = envs[i].reset()
                
                # Print running ASR every 10 steps
                if (total_steps + args.num_processes) % 10 == 0:
                    current_successful = sum(env.successful_attacks for env in envs)
                    current_total = sum(env.total_queries for env in envs)
                    current_asr = current_successful / max(1, current_total)
                    current_avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0.0
                    current_update = total_steps // args.num_steps  # Calculate current update number
                    
                    # Update progress bar with current metrics
                    pbar.set_postfix({
                        'update': f'{current_update}/{num_updates}',
                        'reward': f'{current_avg_reward:.3f}',
                        'ASR': f'{current_asr:.1%}',
                    })
                    
                    tqdm.write(f"[Step {total_steps + args.num_processes}] Running ASR: {current_asr:.2%} ({current_successful}/{current_total})")
                
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
            
            # Logging - Log EVERY update instead of every 10
            avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0.0
            avg_length = np.mean(episode_lengths[-100:]) if episode_lengths else 0.0
            
            # Log to TensorBoard (update-wise, using total_steps as x-axis)
            writer.add_scalar('Training/Episode_Reward', avg_reward, total_steps)
            writer.add_scalar('Training/Episode_Length', avg_length, total_steps)
            writer.add_scalar('Loss/Value_Loss', value_loss, total_steps)
            writer.add_scalar('Loss/Action_Loss', action_loss, total_steps)
            writer.add_scalar('Loss/Entropy', dist_entropy, total_steps)
            
            # Calculate combined ASR across all environments
            total_successful = sum(env.successful_attacks for env in envs)
            total_queries_all = sum(env.total_queries for env in envs)
            combined_asr = total_successful / max(1, total_queries_all)
            
            # Log combined ASR to TensorBoard
            writer.add_scalar('Training/Combined_ASR', combined_asr, total_steps)
            writer.add_scalar('Training/Total_Successful_Attacks', total_successful, total_steps)
            writer.add_scalar('Training/Total_Queries', total_queries_all, total_steps)
            
            # Calculate per-environment ASR
            per_env_asr = []
            per_env_details = []
            for i, env in enumerate(envs):
                env_asr = env.successful_attacks / max(1, env.total_queries)
                per_env_asr.append(env_asr)
                per_env_details.append({
                    'env_id': i,
                    'successful_attacks': env.successful_attacks,
                    'total_queries': env.total_queries,
                    'asr': env_asr
                })
                # Log per-environment ASR to TensorBoard
                writer.add_scalar(f'ASR/Environment_{i}', env_asr, total_steps)
                writer.add_scalar(f'Attacks/Environment_{i}_Successful', env.successful_attacks, total_steps)
                writer.add_scalar(f'Attacks/Environment_{i}_Total', env.total_queries, total_steps)
            
            # Log ASR to separate file EVERY update
            asr_entry = {
                'update': int(update),
                'total_steps': int(total_steps),
                'combined_asr': float(combined_asr),
                'total_successful_attacks': int(total_successful),
                'total_queries': int(total_queries_all),
                'per_environment': per_env_details,
                'avg_reward': float(avg_reward),
                'avg_episode_length': float(avg_length)
            }
            
            if not first_asr_entry:
                asr_log_file.write(',\n')
            json.dump(asr_entry, asr_log_file, indent=2)
            asr_log_file.flush()
            first_asr_entry = False
            
            # Update progress bar with metrics EVERY update
            pbar.set_postfix({
                'update': f'{update}/{num_updates}',
                'reward': f'{avg_reward:.3f}',
                'ASR': f'{combined_asr:.1%}',
                'v_loss': f'{value_loss:.3f}',
                'a_loss': f'{action_loss:.3f}'
            })
            
            # Detailed console logging every log_interval updates
            if update % args.log_interval == 0:
                # Detailed logging
                tqdm.write(f"\n{'='*60}")
                tqdm.write(f"Update {update}/{num_updates} | Steps {total_steps}/{args.num_env_steps}")
                tqdm.write(f"{'='*60}")
                tqdm.write(f"  Avg Reward: {avg_reward:.4f}")
                tqdm.write(f"  Avg Episode Length: {avg_length:.2f}")
                tqdm.write(f"  Combined ASR: {combined_asr:.2%} ({total_successful}/{total_queries_all})")
                tqdm.write(f"  Per-Env ASR: {[f'Env{i}:{rate:.1%}' for i, rate in enumerate(per_env_asr)]}")
                tqdm.write(f"  Value Loss: {value_loss:.4f}")
                tqdm.write(f"  Action Loss: {action_loss:.4f}")
                tqdm.write(f"  Entropy: {dist_entropy:.4f}")
                tqdm.write(f"{'='*60}")
            
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
        
        # Save final model
        final_path = os.path.join(args.save_dir, 'final_model.pt')
        torch.save({
            'model_state_dict': actor_critic.state_dict(),
            'avg_reward': np.mean(episode_rewards[-100:]) if episode_rewards else 0.0,
            'total_steps': total_steps,
        }, final_path)
        
        # Close JSON arrays
        json_file.write('\n]')
        json_file.close()
        asr_log_file.write('\n]')
        asr_log_file.close()
        
        # Close TensorBoard writer
        writer.close()
        print(f"TensorBoard logs saved to {tensorboard_dir}")
        
        # Calculate final ASR statistics
        total_successful = sum(env.successful_attacks for env in envs)
        total_queries_all = sum(env.total_queries for env in envs)
        final_combined_asr = total_successful / max(1, total_queries_all)
        
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        print(f"Run Directory: {run_dir}")
        print(f"Config saved to {config_path}")
        print(f"Final model saved to {final_path}")
        print(f"Training log saved to {json_path}")
        print(f"ASR log saved to {asr_log_path}")
        print(f"\nTotal Steps: {total_steps}")
        print(f"\nFinal Combined ASR: {final_combined_asr:.2%} ({total_successful}/{total_queries_all})")
        print(f"\nPer-Environment Final ASR:")
        for i, env in enumerate(envs):
            env_asr = env.successful_attacks / max(1, env.total_queries)
            print(f"  Env {i}: {env_asr:.2%} ({env.successful_attacks}/{env.total_queries} successful)")
        print("="*60)


if __name__ == '__main__':
    try:
        import multiprocessing
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    main()
