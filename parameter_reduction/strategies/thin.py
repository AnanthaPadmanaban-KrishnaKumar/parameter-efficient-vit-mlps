class ThinMLPStrategy:
    """Reduces ViT MLP hidden dimension by modifying mlp_ratio parameter."""
    
    def __init__(self, target_mlp_ratio=2.0):
        self.mlp_ratio = target_mlp_ratio
        self.applied = False
        self.original_params = None
        self.final_params = None
    
    def get_model_kwargs_modifier(self):
        """Returns kwargs to modify model creation."""
        return {'mlp_ratio': self.mlp_ratio}
    
    def log_stats(self, model, logger):
        """Log parameter reduction statistics."""
        total_params = sum(p.numel() for p in model.parameters())
        
        # ViT-specific MLP inspection
        if hasattr(model, 'blocks'):
            block = model.blocks[0]
            if hasattr(block, 'mlp'):
                d_model = block.mlp.fc1.in_features
                d_ff = block.mlp.fc1.out_features
                num_blocks = len(model.blocks)
                
                mlp_params_per_block = (
                    block.mlp.fc1.weight.numel() + block.mlp.fc1.bias.numel() +
                    block.mlp.fc2.weight.numel() + block.mlp.fc2.bias.numel()
                )
                total_mlp_params = mlp_params_per_block * num_blocks
                
                logger.info("="*60)
                logger.info(f"ThinMLP Strategy Applied:")
                logger.info(f"  MLP ratio: {self.mlp_ratio} ({d_model}→{d_ff}→{d_model})")
                logger.info(f"  Blocks: {num_blocks}")
                logger.info(f"  Total params: {total_params/1e6:.2f}M")
                logger.info(f"  MLP params: {total_mlp_params/1e6:.2f}M")
                logger.info("="*60)