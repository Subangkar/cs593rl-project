"""
Dataset Loader
Handles loading queries and pregenerated responses from CSV files
"""

import os
import csv
import random
import pandas as pd


class DatasetLoader:
    """Handles loading and managing query datasets and pregenerated responses"""
    
    def __init__(self, dataset_path="dataset/prompts_harmful.csv", seed=42):
        """
        Initialize dataset loader
        
        Args:
            dataset_path: Path to the harmful queries CSV file
            seed: Random seed for reproducible sampling
        """
        self.dataset_path = dataset_path
        self.seed = seed
        random.seed(seed)
    
    def load_queries(self, csv_path=None):
        """
        Load queries from CSV file
        
        Args:
            csv_path: Path to CSV file (uses default if None)
            
        Returns:
            List of query strings
        """
        if csv_path is None:
            csv_path = self.dataset_path
        
        queries = []
        
        try:
            # Try pandas first for robust CSV parsing
            df = pd.read_csv(csv_path, header=None)
            queries = df[0].tolist()
        except Exception as e:
            # Fallback to raw text reading if pandas fails
            try:
                with open(csv_path, 'r', encoding='utf-8') as f:
                    queries = [line.strip() for line in f if line.strip()]
            except Exception as e2:
                raise ValueError(f"Failed to load queries from {csv_path}: {e2}")
        
        return queries
    
    def split_train_test(self, queries, train_size=800):
        """
        Split queries into train and test sets
        
        Args:
            queries: List of all queries
            train_size: Number of queries for training (rest is test)
            
        Returns:
            Tuple of (train_queries, test_queries)
        """
        train_queries = queries[:train_size]
        test_queries = queries[train_size:]
        return train_queries, test_queries
    
    def sample_queries(self, queries, frac_samples=1.0, return_indices=False):
        """
        Randomly sample a fraction of queries
        
        Args:
            queries: List of queries to sample from
            frac_samples: Fraction of queries to sample (0.0-1.0)
            return_indices: If True, also return the sampled indices
            
        Returns:
            Sampled queries list, or (sampled_queries, sampled_indices) if return_indices=True
        """
        if frac_samples <= 0.0 or frac_samples > 1.0:
            if return_indices:
                return queries, None
            return queries
        
        if frac_samples == 1.0:
            if return_indices:
                return queries, None
            return queries
        
        original_size = len(queries)
        sample_size = max(1, int(original_size * frac_samples))
        
        # Sample with indices for mapping to pregenerated responses
        all_indices = list(range(len(queries)))
        sampled_indices = random.sample(all_indices, sample_size)
        sampled_indices.sort()  # Keep in order
        sampled_queries = [queries[i] for i in sampled_indices]
        
        if return_indices:
            return sampled_queries, sampled_indices
        return sampled_queries
    
    def load_pregenerated_responses(self, csv_path, queries=None, sampled_indices=None, offset=0):
        """
        Load pregenerated unaligned responses from CSV
        
        Args:
            csv_path: Path to CSV with pregenerated responses
            queries: List of queries to map responses to (optional)
            sampled_indices: If provided, indices of sampled queries for mapping
            offset: Offset for train/test split (0 for train, 800 for test)
            
        Returns:
            Dictionary mapping queries to their pregenerated responses
        """
        if not os.path.exists(csv_path):
            return {}
        
        pregenerated_responses = {}
        
        try:
            # Load all responses in order
            responses_list = []
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    if len(row) >= 2:
                        responses_list.append(row[1])  # Store response only
                    else:
                        responses_list.append('')  # Empty for malformed rows
            
            # Map queries to responses
            if queries is not None:
                if sampled_indices is not None:
                    # Use sampled indices for mapping
                    for local_idx, original_idx in enumerate(sampled_indices):
                        global_idx = original_idx + offset
                        if global_idx < len(responses_list) and local_idx < len(queries):
                            query = queries[local_idx]
                            pregenerated_responses[query] = responses_list[global_idx]
                else:
                    # Use sequential mapping
                    for local_idx, query in enumerate(queries):
                        global_idx = local_idx + offset
                        if global_idx < len(responses_list):
                            pregenerated_responses[query] = responses_list[global_idx]
            
        except Exception as e:
            print(f"Error loading pregenerated responses: {e}")
        
        return pregenerated_responses
    
    def get_dataset_stats(self, queries, pregenerated_responses=None):
        """
        Get statistics about the loaded dataset
        
        Args:
            queries: List of queries
            pregenerated_responses: Dict of pregenerated responses (optional)
            
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'total_queries': len(queries),
            'avg_query_length': sum(len(q) for q in queries) / len(queries) if queries else 0,
            'min_query_length': min(len(q) for q in queries) if queries else 0,
            'max_query_length': max(len(q) for q in queries) if queries else 0,
        }
        
        if pregenerated_responses:
            stats['pregenerated_count'] = len(pregenerated_responses)
            stats['pregenerated_coverage'] = len(pregenerated_responses) / len(queries) if queries else 0.0
        
        return stats
    
    def load_dataset(self, eval=False, frac_samples=1.0, unaligned_csv=None, verbose=True):
        """
        High-level method to load complete dataset with train/test split
        
        Args:
            eval: If True, load test set; if False, load train set
            frac_samples: Fraction of queries to sample
            unaligned_csv: Path to pregenerated responses CSV (optional)
            verbose: If True, print loading information
            
        Returns:
            Tuple of (queries, pregenerated_responses_dict, sampled_indices)
        """
        # Load all queries
        all_queries = self.load_queries()
        
        # Split train/test
        train_queries, test_queries = self.split_train_test(all_queries, train_size=800)
        queries = test_queries if eval else train_queries
        offset = 800 if eval else 0
        
        if verbose:
            split_name = "test" if eval else "train"
            print(f"Loaded {len(queries)} {split_name} queries")
        
        # Sample queries if needed
        sampled_indices = None
        if frac_samples < 1.0:
            queries, sampled_indices = self.sample_queries(queries, frac_samples, return_indices=True)
            if verbose and sampled_indices:
                print(f"Randomly sampled {len(queries)} queries ({frac_samples*100:.1f}%)")
        
        # Load pregenerated responses
        pregenerated_responses = {}
        if unaligned_csv and os.path.exists(unaligned_csv):
            if verbose:
                print(f"Loading pregenerated responses from {unaligned_csv}...")
            pregenerated_responses = self.load_pregenerated_responses(
                unaligned_csv, queries, sampled_indices, offset
            )
            if verbose:
                matched = len(pregenerated_responses)
                total = len(queries)
                if matched > 0:
                    print(f"Mapped {matched}/{total} queries to pregenerated responses ({matched/total*100:.1f}%)")
                else:
                    print("Warning: No pregenerated responses mapped")
        
        return queries, pregenerated_responses, sampled_indices
