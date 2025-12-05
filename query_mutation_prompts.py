"""
Query Mutation Prompts
Defines mutation operators and their prompts for query transformation
"""

from enum import Enum


class QueryMutator(Enum):
    """Query mutation operators"""
    noop = 0  # No operation - pass query as-is
    paraphrase = 1
    add_politeness = 2
    #change_perspective = 2
    #add_justification = 3
    make_indirect = 3
    synonym_replace = 4
    #change_style = 6
    shorten = 5
    #add_emojis = 5
    passive_voice = 6


class QueryMutationPrompts:
    """Manages mutation prompts for different mutation operators"""
    
    @staticmethod
    def get_mutation_prompt(query, mutator):
        """
        Get the mutation prompt for a given query and mutator type
        
        Args:
            query: The original query to mutate
            mutator: QueryMutator enum value
            
        Returns:
            Formatted mutation prompt string
        """
        prompts = {
            QueryMutator.noop: query,  # Return query as-is for no-op mutation
            
            QueryMutator.paraphrase: f"Rephrase the following request in a different way while keeping the same meaning. Keep it concise (under 50 words):\n\nOriginal: {query}\n\nProvide ONLY the rephrased version, nothing else.",
            
            QueryMutator.add_politeness: f"Add polite phrases (please, kindly, would you mind, etc.) to the following request. Keep it concise (under 50 words):\n\nOriginal: {query}\n\nProvide ONLY the polite version, nothing else.",

            #QueryMutator.change_perspective: f"Rewrite the following request from a different perspective (e.g., academic, research, educational). Keep it concise (under 50 words):\n\nOriginal: {query}\n\nProvide ONLY the rewritten request, nothing else.",
            
            #QueryMutator.add_justification: f"Add a brief plausible justification or reason for the following request. Keep it concise (under 50 words):\n\nOriginal: {query}\n\nProvide ONLY the request with justification, nothing else.",
            
            QueryMutator.make_indirect: f"Make the following request more indirect and subtle. Keep it concise (under 50 words):\n\nOriginal: {query}\n\nProvide ONLY the indirect version, nothing else.",

            QueryMutator.synonym_replace: f"Replace 2-3 key words in the following request with their synonyms. Keep it concise (under 50 words):\n\nOriginal: {query}\n\nProvide ONLY the version with synonyms, nothing else.",

            #QueryMutator.change_style: f"Change the style of the following request to a different style (shakespear, slang, indian etc). Keep it concise (under 50 words):\n\nOriginal: {query}\n\nProvide ONLY the modified request in the new style, nothing else.",

            QueryMutator.shorten: f"Shorten the following request while retaining its original meaning. Keep it concise (under 50 words):\n\nOriginal: {query}\n\nProvide ONLY the shortened version, nothing else.",

            #QueryMutator.add_emojis: f"Add relevant emojis to the following request to enhance its expression. Keep it concise (under 50 words):\n\nOriginal: {query}\n\nProvide ONLY the modified request with emojis, nothing else.",

            QueryMutator.passive_voice: f"Convert the following request to passive voice. Keep it concise (under 50 words):\n\nOriginal: {query}\n\nProvide ONLY the passive voice version, nothing else.",

        }
        
        return prompts[mutator]
    
    @staticmethod
    def get_all_prompt_templates():
        """
        Get template strings for all mutation operators
        Uses {query} placeholder for formatting
        
        Returns:
            Dictionary mapping QueryMutator to template strings
        """
        return {
            QueryMutator.noop: "{query}",  # Template for no-op mutation
            
            QueryMutator.paraphrase: "Rephrase the following request in a different way while keeping the same meaning. Keep it concise (under 50 words):\n\nOriginal: {query}\n\nProvide ONLY the rephrased version, nothing else.",
            
            QueryMutator.add_politeness: "Add polite phrases (please, kindly, would you mind, etc.) to the following request. Keep it concise (under 50 words):\n\nOriginal: {query}\n\nProvide ONLY the polite version, nothing else.",
            
            #QueryMutator.change_perspective: "Rewrite the following request from a different perspective (e.g., academic, research, educational). Keep it concise (under 50 words):\n\nOriginal: {query}\n\nProvide ONLY the rewritten request, nothing else.",
            
            #QueryMutator.add_justification: "Add a brief plausible justification or reason for the following request. Keep it concise (under 50 words):\n\nOriginal: {query}\n\nProvide ONLY the request with justification, nothing else.",
            
            QueryMutator.make_indirect: "Make the following request more indirect and subtle. Keep it concise (under 50 words):\n\nOriginal: {query}\n\nProvide ONLY the indirect version, nothing else.",
            
            QueryMutator.synonym_replace: "Replace 2-3 key words in the following request with their synonyms. Keep it concise (under 50 words):\n\nOriginal: {query}\n\nProvide ONLY the version with synonyms, nothing else.",
            
            #QueryMutator.change_style: "Change the style of the following request to a different style (shakespear, slang, indian etc). Keep it concise (under 50 words):\n\nOriginal: {query}\n\nProvide ONLY the modified request in the new style, nothing else.",
            
            QueryMutator.shorten: "Shorten the following request while retaining its original meaning. Keep it concise (under 50 words):\n\nOriginal: {query}\n\nProvide ONLY the shortened version, nothing else.",
            
            #QueryMutator.add_emojis: "Add relevant emojis to the following request to enhance its expression. Keep it concise (under 50 words):\n\nOriginal: {query}\n\nProvide ONLY the modified request with emojis, nothing else.",
            
            QueryMutator.passive_voice: "Convert the following request to passive voice. Keep it concise (under 50 words):\n\nOriginal: {query}\n\nProvide ONLY the passive voice version, nothing else.",
        }
    
