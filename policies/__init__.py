from stable_baselines3.common.policies import register_policy

from .actor_critic import ActorCriticCnnRnnPolicy, ActorCriticPolicy, ActorCriticRnnPolicy

__all__ = [
    "ActorCriticPolicy",
    "ActorCriticRnnPolicy",
    "ActorCriticCnnRnnPolicy",
]

register_policy("MlpPolicy", ActorCriticPolicy)
register_policy("RnnPolicy", ActorCriticRnnPolicy)
register_policy("CnnRnnPolicy", ActorCriticCnnRnnPolicy)
