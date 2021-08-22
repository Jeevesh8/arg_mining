from typing import Optional, Union, List, Tuple
import jax
import jax.numpy as jnp
import haiku as hk

from .utils import add_garbage_dims, remove_garbage_dims


class tree_crf(hk.Module):
    def __init__(
        self,
        prior: Optional[Union[List[List[List[float]]], jnp.ndarray]] = None,
        name: Optional[str] = None,
    ):
        """Constructs a CRF layer that assigns potentials to all different possible trees
        on some sequence of components.

        prior:  An array of shape [n_comps+1, n_comps+1, n_rel_types+1] where (i,j,k)-th entry
                corresponds to the energy predicted by some model for a link from component i
                to component j of type k. The links and the components are 1 indexed and the
                a link to 0-th component from component i, corresponds to i being root node.
                A link of type 0, is None-type link. Usually, the only link with None-type
                will be the link to 0-th component. The prior specifies scaling factors for
                corresponding energies(of the same shape as prior) predicted by the model.

        """
        super(tree_crf, self).__init__(name=name)
        self.prior = prior
        if self.prior is not None:
            raise NotImplementedError(
                "The prior functionality will be implemented in future.")

    @staticmethod
    def _mst(log_energies: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Finds maximal spanning tree for a single sample. See self.mst() for
        detailed documentation.
        """
        M, n_rel_types = log_energies.shape[1], log_energies.shape[2]
        partitions = jnp.eye(M, dtype=jnp.bool_)
        mst_energy = jnp.array([0.0])
        edges = []
        """
        def scan_fn(carry, x):
            _ = x
            mst_energy, log_energies, partitions = carry

            max_index = jnp.unravel_index(jnp.argmax(log_energies),
                                          jnp.shape(log_energies))
            max_energy = log_energies[max_index]
            updatable_sample = jnp.logical_not(jnp.isneginf(max_energy))
            mst_energy += jnp.where(updatable_sample, max_energy, 0.0)
            max_link = jnp.where(
                jnp.stack([updatable_sample] * n_rel_types, axis=-1),
                jnp.squeeze(jnp.transpose(jnp.array(max_index))),
                0.0,
            )

            link_from, link_to, unused_rel_type = max_index
            from_partition = jnp.where(partitions[link_from, :] == 1,
                                       jnp.arange(M), -1)
            to_partition = jnp.where(partitions[link_to, :] == 1,
                                     jnp.arange(M), -1)

            log_energies = add_garbage_dims(log_energies)
            log_energies = jax.ops.index_update(
                log_energies,
                (
                    from_partition,
                    jnp.expand_dims(to_partition, axis=-1),
                    jnp.arange(n_rel_types),
                ),
                -jnp.inf,
            )
            log_energies = jax.ops.index_update(
                log_energies,
                (
                    to_partition,
                    jnp.expand_dims(from_partition, axis=-1),
                    jnp.arange(n_rel_types),
                ),
                -jnp.inf,
            )
            log_energies = remove_garbage_dims(log_energies)

            temp = jnp.logical_or(partitions[link_from, :],
                                  partitions[link_to, :])
            temp_idx = jnp.where(temp, jnp.arange(M), -1)

            partitions = add_garbage_dims(partitions)
            partitions = jax.ops.index_update(
                partitions,
                (jnp.expand_dims(temp_idx, axis=-1), jnp.arange(M)), temp)
            partitions = remove_garbage_dims(partitions)

            return (mst_energy, log_energies, partitions), max_link

        (mst_energy, _,
         _), edges = jax.lax.scan(scan_fn,
                                  init=(mst_energy, log_energies, partitions),
                                  xs=jnp.arange(M),
                                  unroll=M)
        return mst_energy, edges
        """  # FOR-LOOP equivalent
        for _ in range(M):
            max_index = jnp.unravel_index(jnp.argmax(log_energies),
                                          jnp.shape(log_energies))
            max_energy = log_energies[max_index]
            updatable_sample = jnp.logical_not(jnp.isneginf(max_energy))
            mst_energy += jnp.where(updatable_sample, max_energy, 0.0)
            max_link = jnp.where(
                jnp.stack([updatable_sample] * n_rel_types, axis=-1),
                jnp.squeeze(jnp.transpose(jnp.array(max_index))),
                0,
            )

            edges.append(max_link)

            link_from, link_to, unused_rel_type = max_index
            from_partition = jnp.where(partitions[link_from, :] == 1,
                                       jnp.arange(M), -1)
            to_partition = jnp.where(partitions[link_to, :] == 1,
                                     jnp.arange(M), -1)

            log_energies = add_garbage_dims(log_energies)
            log_energies = jax.ops.index_update(
                log_energies,
                (
                    from_partition,
                    jnp.expand_dims(to_partition, axis=-1),
                ),
                -jnp.inf,
            )
            log_energies = jax.ops.index_update(
                log_energies,
                (
                    to_partition,
                    jnp.expand_dims(from_partition, axis=-1),
                ),
                -jnp.inf,
            )
            log_energies = remove_garbage_dims(log_energies)

            temp = jnp.logical_or(partitions[link_from, :],
                                  partitions[link_to, :])
            temp_idx = jnp.where(temp, jnp.arange(M), -1)

            partitions = add_garbage_dims(partitions)
            partitions = jax.ops.index_update(
                partitions,
                (jnp.expand_dims(temp_idx, axis=-1), jnp.arange(M)), temp)
            partitions = remove_garbage_dims(partitions)

        return mst_energy, jnp.stack(edges)

    def mst(self,
            log_energies: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Finds the maximal spanning tree and its score.
        Args:
            log_energies:   A tensor of size [batch_size, M, M, n_rel_types+1] where M = max{1+ (n_comps in i-th sample of batch)}, of the energies of various
                            links predicted by a some model.
        Returns:
        A tuple having:
            1. Tensor of size [batch_size] where the i-th entry denotes cost of MST for i-th sample and

            2. A tensor of size [batch_size, M, 3] where the (i,j)-th 3-sized vector corresponds to the
               j-th edge (link_from, link_to, relation_type) added to the MST of i-th sample. The tensor
               is padded with (0,0,0) jnp.ndarrays.
        """
        return jax.vmap(self._mst)(log_energies)

    @staticmethod
    def _score_tree(log_energies: jnp.ndarray,
                    tree: jnp.ndarray) -> jnp.ndarray:
        """Finds the score of a label tree under the log_energies predicted by some model.
        See self.score_tree() for detailed documentation.
        """
        new_log_energies = jax.ops.index_update(log_energies, (0, 0, 0), 0.0)

        def scan_fn(carry, x):
            edge = x
            score = carry
            return score + new_log_energies[edge[0], edge[1], edge[2]], None

        score, _ = jax.lax.scan(scan_fn, init=jnp.array([0.0]), xs=tree)
        """FOR-LOOP equivalent
        score = jnp.array([0.0])
        for edge in tree:
            score += new_log_energies[edge[0], edge[1], edge[2]]
        """
        return score

    def score_tree(self, log_energies: jnp.ndarray,
                   tree: jnp.ndarray) -> jnp.ndarray:
        """Calculates the log energies of a given batch of trees.
        Args:
            log_energies:   same, as in self.mst()
            tree:           A tensor in the format of second tensor output by self.mst().
        Returns:
            A tensor of size [batch_size] having the score of each tree corresponding to each sample of the batch.
        """
        return jax.vmap(self._score_tree)(log_energies, tree)

    def disc_loss(self, log_energies: jnp.ndarray,
                  label_tree: jnp.ndarray) -> jnp.ndarray:
        """Calculates average loss of a batch of samples.
        Args:
            log_energies:   same, as in self.mst()
            label_tree:     same, as in self.score_tree() [Labels for the actual thread of the tree]
        Returns:
            Average discrimnation loss(loss with partition function estimated with the tree with maximum energy).
        """
        mst_energies, _ = self.mst(log_energies)
        label_tree_scores = self.score_tree(log_energies, label_tree)
        return jnp.mean(mst_energies - label_tree_scores)
