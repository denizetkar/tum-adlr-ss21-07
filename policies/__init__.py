from stable_baselines3.common.policies import register_policy

from .actor_critic import ActorCriticCnnRnnPolicy, ActorCriticRnnPolicy

__all__ = [
    "ActorCriticRnnPolicy",
    "ActorCriticCnnRnnPolicy",
]

register_policy("RnnPolicy", ActorCriticRnnPolicy)
register_policy("CnnRnnPolicy", ActorCriticCnnRnnPolicy)
