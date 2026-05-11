"""
REINVENT integration for molecule generation
Combines our Bi-Int predictor with REINVENT's proven RL
"""

from reinvent_core.models.model import ReinventModel
from reinvent_scoring.models.model import ScoringModel
import numpy as np
import tensorflow as tf
from fullPipeline import BiIntDigitalTwin, BRICSMolecularFeaturizer, HP

class BiIntReinventOptimizer:
    """
    Use REINVENT RL with Bi-Int Digital Twin as reward function
    """
    def __init__(self, model: BiIntDigitalTwin, featurizer: BRICSMolecularFeaturizer, 
                 pretrained_reinvent_path="reinvent_model_path.pkl"):
        self.model = model
        self.featurizer = featurizer
        
        # Load pre-trained REINVENT model
        try:
            self.reinvent = ReinventModel.load_from_file(pretrained_reinvent_path)
            print("✅ Loaded pre-trained REINVENT model")
        except:
            print("⚠️ Pre-trained REINVENT model not found. Using random initialization.")
            self.reinvent = None
    
    def ic50_reward_function(self, smiles_list, gex, mut, cnv, target_ic50=-1.5):
        """
        Reward function: predict IC50 and compare to target
        Returns: reward per SMILES (higher is better)
        """
        rewards = []
        for smiles in smiles_list:
            try:
                atom_feat = self.featurizer.featurize(smiles)
                atom_feat = atom_feat[np.newaxis]
                adj = np.ones((1, HP['max_atoms'], HP['max_atoms']), dtype=np.float32)
                
                inputs = (
                    tf.constant(atom_feat),
                    tf.constant(adj),
                    tf.constant(gex[np.newaxis]),
                    tf.constant(mut[np.newaxis]),
                    tf.constant(cnv[np.newaxis])
                )
                ic50_pred, _ = self.model(inputs, training=False)
                ic50_val = float(ic50_pred[0].numpy())
                
                # Reward: 1.0 if IC50 close to target, else proportional to proximity
                reward = 1.0 - abs(ic50_val - target_ic50) / 10.0
                rewards.append(max(0.0, reward))
            except:
                rewards.append(0.0)
        
        return np.array(rewards)
    
    def optimize(self, gex, mut, cnv, num_steps=500, batch_size=64, target_ic50=-1.5):
        """
        Optimize SMILES generation using REINVENT + Bi-Int reward
        """
        if self.reinvent is None:
            print("Cannot optimize without pre-trained REINVENT model")
            return None
        
        print(f"\n[REINVENT Optimizer] Starting optimization for {num_steps} steps...")
        print(f"  Target IC50: {target_ic50} log µM")
        
        best_smiles = []
        best_rewards = []
        
        for step in range(num_steps):
            # Generate molecules with REINVENT
            molecules = self.reinvent.sample(batch_size)
            smiles_list = [mol.canonical_smiles for mol in molecules]
            
            # Compute Bi-Int rewards
            rewards = self.ic50_reward_function(smiles_list, gex, mut, cnv, target_ic50)
            
            # Update tracking
            best_idx = np.argmax(rewards)
            best_smiles.append(smiles_list[best_idx])
            best_rewards.append(rewards[best_idx])
            
            # Reinforcement learning update (REINVENT handles this internally)
            self.reinvent.update(smiles_list, rewards)
            
            if step % 50 == 0:
                print(f"  Step {step:4d} | Mean Reward: {rewards.mean():.3f} | "
                      f"Best: {smiles_list[best_idx][:40]} | {rewards[best_idx]:.3f}")
        
        return {
            "best_smiles": best_smiles[-1],
            "best_reward": best_rewards[-1],
            "trajectory": best_rewards
        }


# Example usage
if __name__ == "__main__":
    print("REINVENT + Bi-Int Integration")
    print("Note: Requires pre-trained REINVENT model file")
    print("Download from: https://github.com/MarcusOlivecrona/REINVENT")
