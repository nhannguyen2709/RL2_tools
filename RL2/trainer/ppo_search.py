import hydra
import torch.distributed as dist
from tqdm import tqdm
import wandb
from RL2.trainer import Trainer
from RL2.datasets import RLDataset, get_dataloader
from RL2.workers import Actor, Critic
from RL2.workers.agent import Agent
from RL2.utils.algorithms import (
    compute_approx_kl,
    compute_gae,
    compute_reinforce_adv
)
from RL2.utils.comm import initialize_global_process_group
from RL2.utils.logging import time_logger


class PPOSearchTrainer(Trainer):
    """
    PPO Trainer that uses SearchAgent instead of regular Rollout.
    
    This is a minimal extension of PPOTrainer that substitutes
    SearchAgent for the rollout worker.
    """

    def __init__(self, config):
        super().__init__(config)

        self.train_dataloader = self.get_dataloader(True)
        self.test_dataloader = self.get_dataloader(False)

        self.actor = Actor(config.actor, True)
        self.actor.scheduler = self.prepare_scheduler(self.actor)
        if config.actor.kl.coef > 0:
            self.ref_actor = Actor(config.ref_actor, False)
        if config.adv.estimator == "gae":
            self.critic = Critic(config.critic)
            self.critic.scheduler = self.prepare_scheduler(self.critic)
        
        # Use SearchAgent instead of regular Rollout
        self.rollout = Agent(config.rollout)

    def get_dataloader(self, train: bool):

        dataset = RLDataset(
            self.config.data.train_data_path
            if train else self.config.data.test_data_path,
            self.config.data.responses_per_prompt if train else 1
        )

        return get_dataloader(
            dataset,
            self.config.data.prompts_per_rollout
            if train else len(dataset)
        )
    
    @time_logger("compute_approx_kl")
    def compute_approx_kl(self, data_list, step):
        
        kl = 0
        if dist.get_rank() == 0:
            for ex in data_list:
                kl += compute_approx_kl(
                    ex["actor_logps"], ex["ref_logps"], ex["states"]
                )
            kl /= len(data_list)
            wandb.log({"kl_approx": kl}, step=step)

    @time_logger("compute_advantages")
    def compute_advantages(self, data_list, step):

        if self.config.adv.estimator == "gae":
            compute_gae(
                data_list, 
                self.config.adv.gamma, 
                self.config.adv.lamda,
                self.config.adv.norm_var
            )
        elif self.config.adv.estimator == "reinforce":
            compute_reinforce_adv(
                data_list, 
                self.config.adv.norm_var
            )
        else: 
            raise NotImplementedError
            
    def train(self):

        step = self.load_ckpt(
            (self.actor, self.critic)
            if self.config.adv.estimator == "gae"
            else (self.actor,)
        )
        for epoch in range(
            step // len(self.train_dataloader), self.config.trainer.n_epochs
        ):
            for data_list in tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}",
                disable=(dist.get_rank() != 0),
                initial=step % len(self.train_dataloader)
            ):
                step += 1

                data_list = self.rollout(data_list, True, step)

                if self.config.actor.kl.coef > 0:
                    data_list = self.ref_actor.compute_logps(data_list, step)
                if self.config.adv.estimator == "gae":
                    data_list = self.critic.compute_values(data_list, step)
                if self.config.actor.kl.coef > 0 or self.config.actor.update_per_rollout > 1:
                    data_list = self.actor.compute_logps(data_list, step)

                if dist.get_rank() == 0:
                    if self.config.actor.kl.coef > 0:
                        self.compute_approx_kl(data_list, step)
                    self.compute_advantages(data_list, step)

                self.actor.update(data_list, step)
                if self.config.adv.estimator == "gae":
                    self.critic.update(data_list, step)
                self.save_ckpt(
                    (self.actor, self.critic)
                    if self.config.adv.estimator == "gae"
                    else (self.actor,),
                    step
                )

                self.rollout.update(self.actor, step)
                if step % self.config.trainer.test_freq == 0:
                    for data_list in self.test_dataloader:
                        self.rollout(data_list, False, step)

        self.save_model(self.actor)


@hydra.main(config_path="config", config_name="ppo_searchr1", version_base=None)
def main(config):

    initialize_global_process_group()
    trainer = PPOSearchTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()