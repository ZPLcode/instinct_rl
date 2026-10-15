# Source revisions

The training environment is based on
[project-instinct/InstinctLab](https://github.com/project-instinct/InstinctLab),
revision `de6b92f` (Fix symmetric augmentation of base angular velocity).
It includes the local parkour configuration and ray-casting corrections used by
the stair-contact experiments.

The standalone package under `deploy/mujoco` was extracted from the parkour
sim2sim implementation in [InstinctMJ](https://github.com/project-instinct/InstinctMJ).
Its robot assets retain their upstream source and license notices. The Isaac
joint-order mapping follows [Instinct onboard](https://github.com/project-instinct/instinct_onboard),
revision `b6965a4202d8dceaf163dd95edb7742ed5d65e38`.

The tested training dependencies are Isaac Sim 5.1.0, Isaac Lab revision
`f73c331738`, and [Instinct RL](https://github.com/project-instinct/instinct_rl)
revision `687d0ce`.

Original copyright notices, [LICENSE](LICENSE), [CONTRIBUTORS.md](CONTRIBUTORS.md)
and [CONTRIBUTOR_AGREEMENT.md](CONTRIBUTOR_AGREEMENT.md) are retained.
