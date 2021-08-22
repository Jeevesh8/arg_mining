from typing import Optional

import jax
import jax.numpy as jnp
import haiku as hk
"""

-------------------------------------------------A COMMENT ON THE STRUCTURE OF LOG_ENERGIES PRODUCED----------------------------------------------------------

1. The indices going down, to the extreme left indicate a link from that index, and indices along the row at the top indicate a link to that index.

2. No link from 0-th index to any index can be made.(A link to 0-th index denotes the component refers to no other component, i.e., refers = "None")

3. No link from a component to itself, is possible. So all the dash(-)ed positions in the matrix below, have negative infinity in that location for 
all relation types. 

4. If a component is related to no other component, its relation type is autmatically "None". This means that positions marked with (*) have 
negative infinity in all relation types, except "None".

5. If the below matrix A, is to be padded with another matrix, corresponding to another thread, with say 3 components, then that thread's matrix(Matrix B)
will be extended to size 5 and have all the extra(e) positions, filled with negative infinity, for all relation types.

---------Matrix A--------
    0   1   2   3   4   5
0   -   -   -   -   -   -
1   *   -
2   *       -
3   *           -
4   *               -
5   *                   -

---------Matrix B--------
    0   1   2   3   4   5
0   -   -   -   -   e   e
1   *   -           e   e
2   *       -       e   e
3   *           -   e   e
4   e   e   e   e   e   e
5   e   e   e   e   e   e

---------------------------------------------------------------------------------------------------------------------------------------------------------------

"""


class relational_model(hk.Module):
    def __init__(self,
                 n_rels: int,
                 max_comps: int,
                 embed_dim: int,
                 name: Optional[str] = None):
        """Constructs a model with a linear layer to get log energies for links between a number of
        components.

        Args:
            n_rels:     Number of different types of relations that can exist between any two components.
            max_comps:  The maximum number of components in any sample. All samples will be padded to
                        these many components.
            embed_dim:  The dimension of embedding of each component.Assumes fixed size embedding for each
                        component.
        """
        super().__init__(name=name)
        self.n_rels = n_rels
        self.max_comps = max_comps
        self.embed_dim = embed_dim
        self.w = hk.Linear(self.n_rels * self.embed_dim, with_bias=False)

    def _format_log_energies(self, log_energies: jnp.ndarray,
                             pad_mask: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            log_energies:   A [self.max_comps, self.max_comps, self.n_rels] sized array, having the log energy for
                            existence of link from component i to component j of type k at the (i,j,k)-th location.
            pad_mask:       A [self.max_comps] sized array where i-th entry is 1 if the i-th component is an actual
                            one or a component corresponding to root and 0 if it is a padded one.

        Returns:
            Formatted log_energies. As the 0-th component corresponds to connection to root, we need to make sure
            log_energies for connection from 0-th component are not allowed. Moreover, in this formatted log energies
            we set the energies to and from pad components to -infinity. We do the same from links from components
            to themselves.
        """

        log_energies = jnp.where(
            jnp.stack([jnp.diag(jnp.array([1] * self.max_comps))] *
                      self.n_rels,
                      axis=-1),
            -jnp.inf,
            log_energies,
        )

        log_energies = jax.ops.index_update(
            log_energies,
            (0, ),
            -jnp.inf,
        )

        available_from_to = jnp.logical_and(jnp.expand_dims(pad_mask, axis=-1),
                                            pad_mask)

        formatted_log_energies = jnp.where(
            available_from_to, jnp.transpose(log_energies, (2, 0, 1)),
            -jnp.inf)

        return jnp.transpose(formatted_log_energies, (1, 2, 0))

    def _call(self, embds: jnp.ndarray,
              choice_mask: jnp.ndarray) -> jnp.ndarray:
        """Single sample version of self.__call__(). See the same for documentation."""
        num_all_embds = jnp.shape(choice_mask)[0]
        choices = jnp.sort(
            jnp.where(choice_mask, jnp.arange(num_all_embds), num_all_embds))

        indices = jnp.where(choices[:(self.max_comps - 1)]==num_all_embds, -1, choices[:(self.max_comps-1)])+1
        lower_indices = jax.ops.index_update(jnp.roll(indices, shift=(1,), axis=(0,)), (0,), 0)

        from_embds, _ = jax.lax.scan(lambda carry, i: (jnp.where(jnp.expand_dims(jnp.logical_and(lower_indices<=i, i<indices), axis=-1),
                                                                 embds[i]+carry, carry), 0),
                                     init=jnp.zeros((indices.shape[0], embds.shape[-1]), dtype=embds.dtype),
                                     xs=jnp.arange(embds.shape[0]))
        
        from_embds = from_embds/jnp.expand_dims(jnp.where(indices-lower_indices, indices-lower_indices, 1), axis=-1)

        from_embds = jnp.pad(
            from_embds,
            pad_width=((1, 0), (0, 0)),
            constant_values=((1/jnp.shape(embds)[-1], 0), (0, 0)),
        )
        
                
        to_embds = jnp.reshape(self.w(from_embds),
                               (-1, self.embed_dim, self.n_rels))

        log_energies = jnp.dot(from_embds, to_embds)

        pad_mask = jnp.pad(indices != 0,
                           pad_width=(1, 0),
                           constant_values=1)

        return self._format_log_energies(log_energies, pad_mask)

    def __call__(self, embds: jnp.ndarray,
                 choice_mask: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            embds:        A [batch_size, seq_len, embed_dim] sized array having the embeddings of all the
                          words in all sequences in a batch.

            choice_mask:  A [batch_size, seq_len] sized mask, which is 1 for the sequence positions which are to
                          be included as components for relation prediction, and 0 for other positions.
        Returns:
            Log energies array of shape [batch_size, self.max_comps, self.max_comps, self.n_rels] where the (i,j,k,l)-th
            entry corresponds to the log energy of there being a link from component j to component k, of type l in the
            i-th sample. A link to position 0, from component x indicates the log_energy of component x being the root.
        """
        return jax.vmap(self._call)(embds, choice_mask)
