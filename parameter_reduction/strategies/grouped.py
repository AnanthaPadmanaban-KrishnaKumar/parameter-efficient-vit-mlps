import math
import torch
import torch.nn as nn

class GroupedMLPStrategy:
    """Shares MLP parameters between adjacent transformer blocks."""
    
    def __init__(self, scale_factor=0.707):
        """
        Args:
            scale_factor: Scaling for shared weights (0.707 = 1/sqrt(2) for gradient accumulation, 1.0 = no scaling)
        """
        self.scale_factor = scale_factor
        self.grouping = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]
        self.original_params = None
        self.final_params = None
        self.unique_mlps = {}
    
    def apply(self, model):
        """Apply grouped MLP sharing to initialized model."""
        if not hasattr(model, 'blocks'):
            raise ValueError(f"Model does not have 'blocks' attribute. GroupedMLP requires ViT architecture.")
        
        # Record initial params
        self.original_params = sum(p.numel() for p in model.parameters())
        
        # Verify we have 12 blocks (for ViT-B)
        if len(model.blocks) != 12:
            raise ValueError(f"GroupedMLP expects 12 blocks, got {len(model.blocks)}")
        
        # Apply grouping
        for idx_a, idx_b in self.grouping:
            # Scale weights of first MLP in group
            with torch.no_grad():
                model.blocks[idx_a].mlp.fc1.weight.mul_(self.scale_factor)
                model.blocks[idx_a].mlp.fc1.bias.mul_(self.scale_factor)
                model.blocks[idx_a].mlp.fc2.weight.mul_(self.scale_factor)
                # fc2.bias is not scaled (added after multiplication)
            
            # Share the MLP reference
            model.blocks[idx_b].mlp = model.blocks[idx_a].mlp
        
        # Count unique MLPs for verification
        for i in range(12):
            mlp_id = id(model.blocks[i].mlp)
            if mlp_id not in self.unique_mlps:
                self.unique_mlps[mlp_id] = model.blocks[i].mlp
        
        # Record final params
        self.final_params = sum(p.numel() for p in model.parameters())
        
        return model
    
    def log_stats(self, model, logger):
        """Log parameter reduction statistics."""
        if self.original_params is None:
            logger.warning("GroupedMLP strategy not applied yet")
            return
        
        # Calculate MLP params
        mlp_params = sum(p.numel() for mlp in self.unique_mlps.values() for p in mlp.parameters())
        
        logger.info("="*60)
        logger.info(f"GroupedMLP Strategy Applied:")
        logger.info(f"  Grouping pattern: adjacent pairs (0,1), (2,3), ..., (10,11)")
        logger.info(f"  Scale factor: {self.scale_factor:.4f}")
        if self.scale_factor == 1.0:
            logger.info(f"  → No scaling (raw gradient accumulation)")
        elif abs(self.scale_factor - 1.0/math.sqrt(2)) < 0.001:
            logger.info(f"  → 1/√2 scaling (accounts for doubled gradients)")
        logger.info(f"  Unique MLPs: {len(self.unique_mlps)}/12")
        logger.info(f"  MLP parameters: {mlp_params/1e6:.2f}M")
        logger.info(f"  Total params before: {self.original_params/1e6:.2f}M")
        logger.info(f"  Total params after: {self.final_params/1e6:.2f}M")
        logger.info(f"  Parameters saved: {(self.original_params - self.final_params)/1e6:.2f}M")
        logger.info("="*60)