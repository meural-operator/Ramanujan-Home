import math
import random
from .CartesianProductPolyDomain import CartesianProductPolyDomain

class MCTSNode:
    __slots__ = ['assigned_coefs', 'children', 'visits', 'wins']
    def __init__(self, assigned_coefs):
        self.assigned_coefs = assigned_coefs
        self.children = {}
        self.visits = 0
        self.wins = 0.0

    def ucb(self, total_visits, c_param=2.0):
        if self.visits == 0:
            return float('inf')
        # Standard UCB-1 formula
        return (self.wins / self.visits) + c_param * math.sqrt(math.log(total_visits) / self.visits)


class MCTSPolyDomain(CartesianProductPolyDomain):
    """
    Monte Carlo Tree Search (MCTS) PolyDomain
    Maps polynomial coefficients as states in a proper MCTS decision tree.
    Each level of the tree assigns a value to a specific coefficient (a_lead down to b_0).
    
    Uses UCB-1 selection, Progressive Widening for large ranges, random rollouts, 
    and backpropagation to concentrate exploration on bounds that satisfy rigorous 
    asymptotic convergence constraints.
    """
    def __init__(self, a_deg, a_coef_range, b_deg, b_coef_range, mcts_iterations=2000, mcts_top_k=50, c_param=2.0, *args, **kwargs):
        self.mcts_iterations = mcts_iterations
        self.mcts_top_k = mcts_top_k
        self.c_param = c_param
        super().__init__(a_deg, a_coef_range, b_deg, b_coef_range, *args, **kwargs)
        
    def _setup_metadata(self):
        # We run the MCTS pass to dynamically narrow the search domain bounds
        # BEFORE the parent computes iterators and total sizes.
        self._run_mcts_tree()
        super()._setup_metadata()

    def _run_mcts_tree(self):
        """
        Executes the UCB-1 MCTS loop with Progressive Widening.
        Constructs tight bounding boxes from the most successful rolled-out leaves.
        """
        # Flatten ranges for tree ordering: all a_coefs followed by all b_coefs
        all_ranges = self.a_coef_range + self.b_coef_range
        total_depth = len(all_ranges)
        
        root = MCTSNode(assigned_coefs=tuple())
        successful_leaves = []
        
        for _ in range(self.mcts_iterations):
            # 1. SELECT & PROGRESSIVE WIDENING
            node = root
            path = [node]
            depth = 0
            
            while depth < total_depth:
                coef_range = all_ranges[depth]
                range_size = coef_range[1] - coef_range[0] + 1
                
                # Progressive Widening constraint: |Children| <= k * sqrt(N + 1)
                # Limits branching factor dynamically to favor depth over exhaustiveness
                target_children = math.ceil(2.0 * math.sqrt(node.visits + 1))
                target_children = min(target_children, range_size)
                
                if len(node.children) < target_children:
                    # EXPAND: create a new random child branch
                    # Try a few times to find an un-instantiated action
                    for _ in range(50):
                        a = random.randint(coef_range[0], coef_range[1])
                        if a not in node.children:
                            new_coefs = node.assigned_coefs + (a,)
                            child = MCTSNode(assigned_coefs=new_coefs)
                            node.children[a] = child
                            node = child
                            path.append(node)
                            depth += 1
                            break
                    break  # Stop tree traversal and drop into rollout
                else:
                    # SELECT: pick existing child with max UCB
                    best_child = None
                    best_score = -float('inf')
                    for child in node.children.values():
                        score = child.ucb(node.visits, self.c_param)
                        if score > best_score:
                            best_score = score
                            best_child = child
                    
                    if best_child is None:
                        break  # Failsafe
                        
                    node = best_child
                    path.append(node)
                    depth += 1
            
            # 2. ROLLOUT (Random Simulation to leaf)
            rollout_coefs = list(node.assigned_coefs)
            for d in range(depth, total_depth):
                r = all_ranges[d]
                rollout_coefs.append(random.randint(r[0], r[1]))
                
            # 3. EVALUATE LEAF (Constraints Check)
            a_len = len(self.a_coef_range)
            a_c = rollout_coefs[:a_len]
            b_c = rollout_coefs[a_len:]
            
            reward = 0.0
            # filter_gcfs internally checks is_asymptotically_convergent (Poincaré limits)
            if self.filter_gcfs(a_c, b_c):
                reward = 1.0
                successful_leaves.append((a_c, b_c))
                    
            # 4. BACKPROPAGATE
            for n in path:
                n.visits += 1
                n.wins += reward
                
        # 5. AGGREGATE BOUNDS
        if len(successful_leaves) > 0:
            # Drop duplicates and take top K valid results
            unique_leaves = list(set(tuple(a) + tuple(b) for a, b in successful_leaves))
            unique_leaves = unique_leaves[:self.mcts_top_k]
            
            a_len = len(self.a_coef_range)
            success_a = [list(leaf[:a_len]) for leaf in unique_leaves]
            success_b = [list(leaf[a_len:]) for leaf in unique_leaves]
            
            # Compute new min/max bounding box covering the successful samples
            for idx in range(len(self.a_coef_range)):
                min_a, max_a = min([a[idx] for a in success_a]), max([a[idx] for a in success_a])
                self.a_coef_range[idx] = [min_a, max_a]
                
            for idx in range(len(self.b_coef_range)):
                min_b, max_b = min([b[idx] for b in success_b]), max([b[idx] for b in success_b])
                self.b_coef_range[idx] = [min_b, max_b]

