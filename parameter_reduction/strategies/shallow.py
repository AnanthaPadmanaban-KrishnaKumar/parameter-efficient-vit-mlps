import torch
import torch.nn as nn

class ShallowMLPStrategy:
    """Reduces ViT MLP width by pruning after full initialization."""
    
    def __init__(self, target_ratio=0.5):
        """
        Args:
            target_ratio: Fraction of hidden dim to keep (0.5 = prune from 3072 to 1536)
        """
        self.target_ratio = target_ratio
        self.original_params = None
        self.final_params = None
        self.pruned_dims = {}
    
    def apply(self, model):
        """Apply shallow MLP pruning to initialized model."""
        if not hasattr(model, 'blocks'):
            raise ValueError(f"Model does not have 'blocks' attribute. ShallowMLP requires ViT architecture.")
        
        with torch.no_grad():
            # Record initial params
            self.original_params = sum(p.numel() for p in model.parameters())
            
            # Get dimensions from first block
            first_block = model.blocks[0]
            full_dim = first_block.mlp.fc1.out_features
            target_dim = int(full_dim * self.target_ratio)
            
            # Store for logging
            self.pruned_dims = {
                'original': full_dim,
                'target': target_dim,
                'ratio': self.target_ratio
            }
            
            # Apply pruning to each block
            for i, block in enumerate(model.blocks):
                # Prune fc1: (768, 3072) -> (768, 1536)
                old_weight_fc1 = block.mlp.fc1.weight.data
                old_bias_fc1 = block.mlp.fc1.bias.data
                block.mlp.fc1.weight = nn.Parameter(old_weight_fc1[:target_dim, :].clone())
                block.mlp.fc1.bias = nn.Parameter(old_bias_fc1[:target_dim].clone())
                block.mlp.fc1.out_features = target_dim
                
                # Prune fc2: (3072, 768) -> (1536, 768)
                old_weight_fc2 = block.mlp.fc2.weight.data
                block.mlp.fc2.weight = nn.Parameter(old_weight_fc2[:, :target_dim].clone())
                block.mlp.fc2.in_features = target_dim
            
            # Record final params
            self.final_params = sum(p.numel() for p in model.parameters())
        
        return model
    
    def log_stats(self, model, logger):
        """Log parameter reduction statistics."""
        if self.original_params is None:
            logger.warning("ShallowMLP strategy not applied yet")
            return
        
        num_blocks = len(model.blocks) if hasattr(model, 'blocks') else 0
        
        logger.info("="*60)
        logger.info(f"ShallowMLP Strategy Applied:")
        logger.info(f"  Preserved 86M initialization, then pruned")
        logger.info(f"  MLP hidden dim: {self.pruned_dims['original']}→{self.pruned_dims['target']}")
        logger.info(f"  Reduction ratio: {self.target_ratio}")
        logger.info(f"  Blocks modified: {num_blocks}")
        logger.info(f"  Initial params (86M init): {self.original_params/1e6:.2f}M")
        logger.info(f"  Final params after pruning: {self.final_params/1e6:.2f}M")
        logger.info(f"  Parameters removed: {(self.original_params - self.final_params)/1e6:.2f}M")
        logger.info("="*60)