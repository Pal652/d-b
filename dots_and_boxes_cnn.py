# dots_ai_full_with_domino.py
# Full integrated Dots & Boxes: Option B value-net + MCTS + heuristics + domino decisions + Pygame UI
# Save as dots_ai_full_with_domino.py
# Requires: numpy, torch, pygame

import argparse
import copy
import math
import os
import random
import sys
from collections import deque
from typing import List, Tuple, Dict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import functools
import hashlib

import pygame

def GetPath():
    return Path(__file__).parent.absolute()

# -------------------------
# Environment: Table
# -------------------------
class Table:
    """
    Clean Dots and Boxes board representation using lowercase move tags 'h'/'v'.
    """

    def __init__(self, N, UI:bool=False):
        self.N = N  # boxes per row/col
        # Horizontal edges: (N+1) x N
        self.horizontal = np.zeros((N+1, N), dtype=np.int8)
        # Vertical edges: N x (N+1)
        self.vertical = np.zeros((N, N+1), dtype=np.int8)

        self.remaining_boxes = N * N

        # Scores for players (Player1 / Player2). We keep separate counters.
        #self.scoreA = 0
        #self.scoreB = 0
        self.score = 0 # Player1 score - Player 2 score

        # True => Player1 to move, False => Player2 to move
        self.FirstPlayer = True

        self.owner = None
        if (UI): self.owner = np.zeros((N, N), dtype=np.int8)

    # ---------------------------
    # Box helpers
    # ---------------------------
    def box_filled_count(self, bx: int, by: int) -> int:
        """Return number of filled edges of box (bx,by) in 0..4."""
        c = 0
        if self.horizontal[by, bx] == 1: c += 1  # top
        if self.horizontal[by+1, bx] == 1: c += 1  # bottom
        if self.vertical[by, bx] == 1: c += 1  # left
        if self.vertical[by, bx+1] == 1: c += 1  # right
        return int(c)

    def box_empty_edges(self, bx: int, by: int):
        """
        Return list of empty edge indices (0..3) for the box (same convention used
        elsewhere):
         0 = top -> ('h', bx, by)
         1 = right -> ('v', bx+1, by)
         2 = bottom -> ('h', bx, by+1)
         3 = left -> ('v', bx, by)
        """
        empties = []
        if self.horizontal[by, bx] == 0:
            empties.append(0)
        if self.vertical[by, bx+1] == 0:
            empties.append(1)
        if self.horizontal[by+1, bx] == 0:
            empties.append(2)
        if self.vertical[by, bx] == 0:
            empties.append(3)
        return empties

    def edge_boxes(self, edge: Tuple[str,int,int]): # edge to boxes
        """
        Given an edge ('h'/'v', x, y) return list of boxes that share it.
        Order: list of (bx,by). Boundary edges return a single box.
        """
        etype, x, y = edge
        N = self.N
        boxes = []
        et = etype.lower()
        if et == 'h':
            # horizontal edge at (x,y) is top of box (x,y) and bottom of box (x,y-1)
            # top box: (x, y) if 0 <= y < N
            if 0 <= y < N:
                boxes.append((x, y))
            # bottom box: (x, y-1) if 1 <= y <= N
            if 0 < y <= N:
                boxes.append((x, y-1))
        elif et == 'v':
            # vertical edge at (x,y) is left of box (x,y) and right of box (x-1,y)
            # left box: (x, y) if 0 <= x < N
            if 0 <= x < N:
                boxes.append((x, y))
            # right box: (x-1, y) if 1 <= x <= N
            if 0 < x <= N:
                boxes.append((x-1, y))
        else:
            raise ValueError("edge type must be 'h' or 'v'")
        return boxes

    def legal_moves(self) -> List[Tuple[str,int,int]]:
        moves = []
        # horizontals: 'h'
        for y in range(self.N+1):
            for x in range(self.N):
                if self.horizontal[y, x] == 0:
                    moves.append(('h', x, y))
        # verticals: 'v'
        for y in range(self.N):
            for x in range(self.N+1):
                if self.vertical[y, x] == 0:
                    moves.append(('v', x, y))
        return moves

    def apply_move(self, move: Tuple[str,int,int]) -> int:
        """
        Apply move ('h'/'v', x, y). Returns the number of boxes completed by this move (0,1,2).
        Updates scores and player turn accordingly.
        """

        etype, x, y = move
        et = etype.lower()
        if et == 'h':
            if self.horizontal[y, x] == 1:
                raise ValueError("horizontal edge already filled")
            self.horizontal[y, x] = 1
        elif et == 'v':
            if self.vertical[y, x] == 1:
                raise ValueError("vertical edge already filled")
            self.vertical[y, x] = 1
        else:
            raise ValueError("move must start with 'h' or 'v'")

        # Count boxes completed by this move (boxes that now have filled_count == 4)
        completed = 0
        for (bx, by) in self.edge_boxes((et, x, y)):
            if self.box_filled_count(bx, by) == 4:
                if (self.owner is not None): self.owner[by, bx] = int(self.FirstPlayer) + 1
                completed += 1

        # Award points and update player turn
        if completed > 0:
            if self.FirstPlayer:    self.score += completed
            else:                   self.score -= completed
            self.remaining_boxes -= completed
            # current player continues
        else:
            # switch turn
            self.FirstPlayer = not self.FirstPlayer

        return int(completed)

    def clone(self):
        t = Table(self.N)
        t.horizontal = self.horizontal.copy()
        t.vertical = self.vertical.copy()
        t.score = int(self.score)
        t.remaining_boxes = int(self.remaining_boxes)
        t.FirstPlayer = bool(self.FirstPlayer)
        return t

    def game_over(self):
        return self.remaining_boxes == 0

    def max_score(self):
        return self.N * self.N
    
    def __hash__(self):
        return hash((self.horizontal.tobytes(), self.vertical.tobytes()))





def encode_table(t: Table) -> np.ndarray:
    N = t.N
    H = np.zeros((N+1, N+1), dtype=np.float32)
    V = np.zeros((N+1, N+1), dtype=np.float32)
    H[:, :N] = t.horizontal
    V[:N, :] = t.vertical
    return np.stack([H, V], axis=0)  # shape (2, N+1, N+1)


# -------------------------
# Box helpers (edge indexing, mapping to moves)
# -------------------------
def box_empty_edges(table: Table, bx: int, by: int): # return unfilled edges of a box
    N = table.N
    empties = []
    # top
    if table.horizontal[by, bx] == 0:
        empties.append(0)
    # right
    if table.vertical[by, bx+1] == 0:
        empties.append(1)
    # bottom
    if table.horizontal[by+1, bx] == 0:
        empties.append(2)
    # left
    if table.vertical[by, bx] == 0:
        empties.append(3)
    return empties

def box_filled_count(table: Table, bx: int, by: int):
    return 4 - len(box_empty_edges(table, bx, by))

def neighbor_box_by_edge(bx: int, by: int, edge_idx: int, N: int): # next box in given direction
    if edge_idx == 0:
        return (bx, by-1) if by-1 >= 0 else None
    elif edge_idx == 1:
        return (bx+1, by) if bx+1 < N else None
    elif edge_idx == 2:
        return (bx, by+1) if by+1 < N else None
    else:
        return (bx-1, by) if bx-1 >= 0 else None

def box_edge_to_move(bx:int, by:int, edge_idx:int): # format: ('h'|'v', x, y)
    if edge_idx == 0:
        return ('h', bx, by)
    elif edge_idx == 1:
        return ('v', bx+1, by)
    elif edge_idx == 2:
        return ('h', bx, by+1)
    else:
        return ('v', bx, by)

# -------------------------
# Component detection (broken chains/loops only)
# -------------------------
def detect_chainloop_components(table: Table): # broken chains/loops
    """
    Detect broken chains and broken loops according to the simplified rules:
      - Broken chain: 3-filled → (sequence of 2-filled) → non 2/3-filled (not included)
      - Broken loop: 3-filled → (sequence of 2-filled) → 3-filled (included)
    Returns list of components: {'type':'chain'|'loop','sequence':[(bx,by),...]}
    """
    N = table.N
    visited = set()
    comps = []

    # Precompute empties and filled counts
    empty_edges = {}
    filled_counts = {}
    for by in range(N):
        for bx in range(N):
            empt = box_empty_edges(table, bx, by)
            empty_edges[(bx, by)] = empt
            filled_counts[(bx, by)] = 4 - len(empt)

    for by in range(N):
        for bx in range(N):
            if filled_counts[(bx, by)] != 3 or (bx, by) in visited:
                continue

            # start new component
            seq = [(bx, by)]
            visited.add((bx, by))
            prev = (bx, by)
            empties = empty_edges[(bx, by)]
            if not empties:
                comps.append({'type': 'chain', 'sequence': seq})
                continue
            start_edge = empties[0]
            curr = neighbor_box_by_edge(bx, by, start_edge, N)
            is_loop = False

            while curr is not None:
                cbx, cby = curr
                if (cbx, cby) in visited:
                    break
                cnt = filled_counts.get((cbx, cby), 0)

                if cnt == 2:
                    seq.append((cbx, cby))
                    visited.add((cbx, cby))
                    # continue along the edge that is not pointing back to prev
                    curr_empties = empty_edges[(cbx, cby)]
                    back_edge = None
                    for e in curr_empties:
                        if neighbor_box_by_edge(cbx, cby, e, N) == prev:
                            back_edge = e
                            break
                    cont_edge = None
                    for e in curr_empties:
                        if e != back_edge:
                            cont_edge = e
                            break
                    if cont_edge is None:
                        break  # chain ends
                    prev = (cbx, cby)
                    curr = neighbor_box_by_edge(cbx, cby, cont_edge, N)
                    continue

                elif cnt == 3:
                    # broken loop: include terminal
                    seq.append((cbx, cby))
                    visited.add((cbx, cby))
                    is_loop = True
                    break

                else:
                    # broken chain ends here; do not include terminal
                    break

            comps.append({'type': 'loop' if is_loop else 'chain', 'sequence': seq})

    return comps

def detect_two_filled_components(table: Table):
    """
    Find connected components of boxes that have exactly 2 filled edges.
    Return list of {'type':'open_chain'|'open_loop', 'sequence':[(bx,by), ...]}.
    """
    N = table.N
    # compute filled counts
    filled_counts = {}
    for by in range(N):
        for bx in range(N):
            filled_counts[(bx,by)] = box_filled_count(table, bx, by)

    # nodes with exactly 2 filled edges
    nodes = [(bx,by) for by in range(N) for bx in range(N) if filled_counts[(bx,by)] == 2 or filled_counts[(bx,by)] == 3]
    node_set = set(nodes)
    if not nodes:
        return []

    # adjacency: boxes connected if they share an empty edge
    adj = {n: set() for n in nodes}
    for (bx,by) in nodes:
        empt = box_empty_edges(table, bx, by)
        for e in empt:
            nb = neighbor_box_by_edge(bx, by, e, N)
            if nb and nb in node_set:
                adj[(bx,by)].add(nb)

    visited = set()
    comps = []
    for node in nodes:
        if node in visited: continue
        # BFS to gather component
        stack = [node]; comp_nodes = []
        visited.add(node)
        while stack:
            cur = stack.pop()
            comp_nodes.append(cur)
            #if (filled_counts[cur] == 3): continue # poison skip
            for nb in adj[cur]:
                if nb not in visited:
                    visited.add(nb); stack.append(nb)
        # degrees inside component
        degrees = [len(adj[n]) for n in comp_nodes]
        if all(d == 2 for d in degrees):
            # closed cycle -> open_loop
            comps.append({'type':'open_loop', 'sequence': comp_nodes})
        else:
            # path: order nodes from endpoint (deg==1) if possible
            endpoints = [n for n in comp_nodes if (len(adj[n]) == 1)]
            if endpoints:
                if (filled_counts[endpoints[0]] == 3 or filled_counts[endpoints[1]] == 3): # poison chain with 3 filled
                    continue # broken shit

                start = endpoints[0]
                seq = []
                prev = None
                cur = start
                while True:
                    seq.append(cur)
                    nexts = [nb for nb in adj[cur] if nb != prev]
                    if not nexts:
                        break
                    prev = cur
                    cur = nexts[0]
                    if cur == start:
                        break
                comps.append({'type':'open_chain', 'sequence': seq})
            elif (filled_counts[comp_nodes[0]] == 3):
                pass # single 3-edged box
            else:
                comps.append({'type':'open_chain', 'sequence': comp_nodes})

    #print(comps)
    return comps

# -------------------------
# Generate root moves with collapse (improved: collapse open 2-filled components)
# -------------------------
def generate_root_moves_with_collapse(state: Table):
    """
    Build root moves with:
      - suicidal filtering (if any safe moves exist only those are used)
      - remove the edges which do the same thing
    """

    # remove all two filled compoment moves (any moove, that makes a 3 edged shit)
    # add back the opening moves
    # return an "ornage" and "red" list for UI


    legal = state.legal_moves()
    # Step 1: safe moves

    

    #move:
        #sacrify
            #suicidal
            #not suicidal (true sacryfy)
        #safe
    
    safe_moves = []
    TrueSacryfie_moves = []
    suicidal_moves = []

    for mv in legal:
        if not is_sacrify_move(state, mv): safe_moves.append(mv)
        else:       
            if is_suicidal_move(state, mv): suicidal_moves.append(mv)
            else:                           TrueSacryfie_moves.append(mv)                    
        


    have_safe = bool(safe_moves) or bool(TrueSacryfie_moves) # there are non suicidal moves


    # all suicidal, so any moove is fine COLLAPSING STILL NESSESARY

    use_moves = safe_moves


    # Step 2: detect open (two-filled) components and prepare collapse
    collapsed_actions = []
    collapse_candidates = detect_two_filled_components(state)
    for c in collapse_candidates:
        seq = c['sequence']
        bx0, by0 = seq[0]
        # collect all free edges inside this component

        if (have_safe and len(seq) > 2): continue # long ones are suboptimal
        
        # choose good edge in this component
        
        e0 = box_empty_edges(state, bx0, by0)[0]
        mv = box_edge_to_move(bx0, by0, e0)

        if (len(seq) != 2):
            collapsed_actions.append(mv)
            continue
        
        # 2 long chain
        if (mv not in TrueSacryfie_moves):
            e0 = box_empty_edges(state, bx0, by0)[1]
            mv = box_edge_to_move(bx0, by0, e0)

        
        collapsed_actions.append(mv) # add back the chain/loopopening move

    # add abstract collapsed actions
    use_moves.extend(collapsed_actions)

    if not use_moves:
        return [], [], []
        #raise Exception()
    
    return use_moves, suicidal_moves, TrueSacryfie_moves


def domino_moves(table: Table, comp): # gives the devour and decline moves 
    """
    For the component, generate two candidate moves:
    - return the two moves that devour or decline the component (in random order)
    """
    seq = comp['sequence']

    # devour move: first box's empty edge
    bx0, by0 = seq[1]
    empt0 = box_empty_edges(table, bx0, by0)
    move1 = box_edge_to_move(bx0, by0, empt0[0])
    move2 = box_edge_to_move(bx0, by0, empt0[1])

    return [move1, move2]


# -------------------------
# Heuristic (updated priority order + domino detection)
# -------------------------

def move_from_comp(table:Table, comp): # eat 1 from chain/loop move
    bx, by = comp['sequence'][0]
    empt = box_empty_edges(table, bx, by)
    if not empt: return None # game ended
    return box_edge_to_move(bx, by, empt[0])

def heuristic_forced_move(table: Table):
    comps = detect_chainloop_components(table)

    chains = [c for c in comps if c['type']=='chain']
    loops  = [c for c in comps if c['type']=='loop']

    # Priority rules (broken chains/loops >2)
    for comp in chains:
        if len(comp['sequence']) != 2:
            mv = move_from_comp(table, comp)
            if mv: return mv

    for comp in loops:
        if len(comp['sequence']) != 4:
            mv = move_from_comp(table, comp)
            if mv: return mv

    # Handle multiple loops/chains
    if len(loops) >= 2:
        mv = move_from_comp(table, loops[0])
        if mv: return mv

    if len(chains) >= 2:
        mv = move_from_comp(table, chains[0])
        if mv: return mv

    if chains and loops:
        mv = move_from_comp(table, loops[0])
        if mv: return mv

    # Domino detection: len==2 chain OR len==4 loop
    domino_comps = [c for c in comps if (c['type']=='chain' and len(c['sequence']) == 2) or
                                     (c['type']=='loop' and len(c['sequence']) == 4)]
    if domino_comps:
        comp = domino_comps[0] # should be len 1 btw
        d_moves = domino_moves(table, comp)
        #print("Devour:", dev, "Decline:", dec)
        return {'domino': d_moves}

    return None



# -------------------------
# Suicidal move detection & collapsed-move generation
# -------------------------
def _comp_sig(comp):
    """Canonical signature for a component for comparison."""
    return (comp['type'], tuple(comp['sequence']))

def is_suicidal_move(table: Table, move):
    """
    A move is suicidal if:
      - It does NOT give the player points (no boxes completed)
      - AND it creates a NEW broken chain of length >=2
        OR a NEW broken loop of length >=4.
    Moves that score points are NEVER suicidal, because the player moves again.
    """

    score_before = table.score  # (P1 - P2)

    # compute components before move
    before = set(_comp_sig(c) for c in detect_chainloop_components(table))

    # clone and apply move
    t = table.clone()
    t.apply_move(move)


    # If player gained points -> NOT suicidal
    score_after = t.score
    if score_after != score_before: return False


    # Now check for new dangerous components
    after = set(_comp_sig(c) for c in detect_chainloop_components(t))
    new = after - before

    for typ, seq in new:
        if typ == 'chain' and len(seq) >= 2:
            return True
        if typ == 'loop' and len(seq) >= 4:
            return True

    return False


def is_sacrify_move(table: Table, move):
    """
    move gives a 3 edged box
    """

    ebs = table.edge_boxes(move)
    if any(table.box_filled_count(*eb) == 2 for eb in ebs): return True

    return False


# -------------------------
# Value network (Option B: absolute Player1 perspective)
# -------------------------

"""
class DotsValueNet(nn.Module):
    def __init__(self, board_size, channels=2):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = self.pool(h).view(h.size(0), -1)
        h = F.relu(self.fc1(h))
        v = torch.tanh(self.fc2(h))
        return v.squeeze(-1)
"""

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(h + x)


class DotsValueNet(nn.Module):
    def __init__(self, board_size, input_channels=2, channels=64, blocks=6):
        super().__init__()
        self.conv_in = nn.Conv2d(input_channels, channels, 3, padding=1)
        self.bn_in = nn.BatchNorm2d(channels)

        self.resblocks = nn.Sequential(*[
            ResBlock(channels) for _ in range(blocks)
        ])

        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(channels, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        h = F.relu(self.bn_in(self.conv_in(x)))
        h = self.resblocks(h)
        h = self.pool(h).view(h.size(0), -1)
        h = F.relu(self.fc1(h))
        return torch.tanh(self.fc2(h)).squeeze(-1)

# -------------------------
# Replay buffer
# -------------------------
class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buf = deque(maxlen=capacity) # (encoded states, target values)

    def push(self, s: np.ndarray, target: float):
        self.buf.append((s.astype(np.float32), float(target)))

    def sample(self, n):
        n = min(n, len(self.buf))
        batch = random.sample(self.buf, n) # gives n game X, Y-s
        states = np.stack([b[0] for b in batch], axis=0)
        targets = np.array([b[1] for b in batch], dtype=np.float32)
        return states, targets

    def __len__(self):
        return len(self.buf)

# -------------------------
# MCTS (value-net leaf eval) - unchanged core, used only if no forced move or domino-select
# -------------------------

class BatchEvaluator:
    def __init__(self, model, device, batch_size=32):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.pending = []  # list of (node, state, path)

    def queue(self, node, path):
        """
        Add a leaf for later batch evaluation.
        Returns True if evaluation has been queued,
        False if batch is full and must be processed immediately.
        """
        self.pending.append((node, list(path)))
        return len(self.pending) < self.batch_size

    def flush(self):
        if not self.pending:
            return []

        # Encode all tables at once
        arr = np.stack([encode_table(node.state) for (node, _) in self.pending], axis=0)
        inp = torch.from_numpy(arr).to(self.device)

        with torch.no_grad():
            vals = self.model(inp).cpu().numpy()

        self.pending # debub

        results = []
        # node is NOT path[-1][0] btw
        for ((node, path), v) in zip(self.pending, vals):
            # convert to perspective-of-root-leaf: value is from the viewpoint of st.FirstPlayer
            val = float(v) if node.FirstPlayer else -float(v)
            results.append((node, val, path))

        self.pending.clear()
        return results


import math, numpy as np
from typing import Dict, Tuple, List
import torch
import torch.nn as nn


class MCTS_inner_Node:
    def __init__(self, state:Table):
        self._state = state
        self.is_expanded = False

        # P, N, W for each move
        self.priors: Dict[Tuple[str,int,int], float] = {}
        self.visit_count: Dict[Tuple[str,int,int], int] = {}
        self.value_sum: Dict[Tuple[str,int,int], float] = {}

        self.total_visits = 0 # N(s)

class MCTSNode:
    def __init__(self):
        # call from_node or from_table
        self.node = None
        self.FirstPlayer = None
        self.Score = None

        # child move → node
        self.children: Dict[Tuple[str,int,int], "MCTSNode"] = {}

    def from_node(self, node:MCTS_inner_Node, FirstPlayer, Score): # same state as an already checked one.
        self.node = node
        self.FirstPlayer = FirstPlayer
        self.Score = Score
        return self

    def from_table(self, table:Table): # new state
        self.node = MCTS_inner_Node(table)
        self.FirstPlayer = table.FirstPlayer
        self.Score = table.score
        return self
    

    # kills paralell running :(
    @property
    def state(self):
        self.node._state.FirstPlayer = self.FirstPlayer # to make functions work properly (like applymove)
        self.node._state.score = self.Score
        return self.node._state
    
    def apply_move_to_node(self, move): # only do, if you want to MOVE the node in the tree (for exaple making a forced move, and seting this node to its child)
        self.state.apply_move(move)
        self.FirstPlayer = self.node._state.FirstPlayer
        self.Score = self.node._state.score
        # CHANGES HASH


class MCTS:
    def __init__(self, model:nn.Module, device,
                 num_simulations:int = 50, c_puct:float = 1.0,
                 dirichlet_alpha:float = 0.3, dir_noise_eps:float = 0.25):

        self.model = model
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dir_noise_eps = dir_noise_eps

        self.batcher = BatchEvaluator(model, device, batch_size=1)#round(num_simulations*0.08))

        self.nodes: Dict[int, MCTS_inner_Node] = {} # table hash to MCTSNode


    # ----------------------------------------------------------
    # Utility: expand a node
    # ----------------------------------------------------------

    def expand(self, node:MCTSNode):

        # domino and forced filter for MCTS deeper moves
        fm = heuristic_forced_move(node.state)
        if isinstance(fm, dict) and 'domino' in fm: moves = fm['domino']
        elif isinstance(fm, tuple): moves = [fm]
        else: moves, _, _ = generate_root_moves_with_collapse(node.state)

        if not moves:
            node.node.is_expanded = True
            return

        p = 1.0 / len(moves)
        for mv in moves:
            node.node.priors[mv] = p
            node.node.visit_count[mv] = 0
            node.node.value_sum[mv] = 0.0

        node.node.is_expanded = True


    # ----------------------------------------------------------
    # Utility: add Dirichlet noise at root
    # ----------------------------------------------------------
    def add_root_noise(self, node:MCTSNode):
        moves = list(node.node.priors.keys())
        if not moves: 
            return

        noise = np.random.dirichlet([self.dirichlet_alpha]*len(moves))
        for mv, eps in zip(moves, noise):
            old = node.node.priors[mv]
            node.node.priors[mv] = old*(1-self.dir_noise_eps) + eps*self.dir_noise_eps


    # ----------------------------------------------------------
    # Utility: PUCT move selection
    # ----------------------------------------------------------
    def select_move(self, node:MCTSNode):
        best = -1e9
        best_mv = None

        sqrt_parent = math.sqrt(max(1, node.node.total_visits))

        for mv, P in node.node.priors.items():
            N = node.node.visit_count[mv]
            Q = (node.node.value_sum[mv] / N) if N > 0 else 0.0
            u = Q + self.c_puct * P * sqrt_parent / (1 + N)
            if u > best:
                best = u
                best_mv = mv

        return best_mv


    # ----------------------------------------------------------
    # Utility: NN evaluation for leaf
    # ----------------------------------------------------------
    def evaluate_leaf(self, node: MCTSNode, path:List[Tuple[MCTSNode, Tuple[str,int,int]]]):
        queued = self.batcher.queue(node, path)
        if not queued:
            # batch full -> flush immediately and return results for processing
            return self.batcher.flush()
        return []
    
    #"""
    def PropBack(self, to_prop, leaf: MCTSNode, path:List[Tuple[MCTSNode, Tuple[str,int,int]]]):

        #to_prop = 0 # debug test MCTS

        # in the tree every value is from the viewpoint of the NodePlayer,
        # and the Neural network gives the value from the viewpoint of the NodePlayer

        # back to FP pov (neural network deosent now who comes first, so it is node player pov)
        if (not leaf.FirstPlayer): to_prop = -to_prop # to FP pov (for unwrapping to full_game_score_pred)

        # get Score_pred of nn for state from value ((Score_end-Score_node)/Boxes_left) - backwards
        full_game_score_pred = to_prop
        full_game_score_pred *= leaf.state.remaining_boxes # total delta_score
        full_game_score_pred += leaf.Score # pred points
        
        for node, action in reversed(path):
            # node goes from        ] leaf ; root ]
            
            # root to lef points + pred points
            if (node.state.remaining_boxes == 0): continue

            # wrap back, and convert to node player pov
            nodePov = ((full_game_score_pred - node.Score) / node.state.remaining_boxes)
            if (not node.FirstPlayer): nodePov = -nodePov


            node.node.value_sum[action] += nodePov


            #print("backprop leaf, to_prop:", to_prop, "pathlen:", len(path)


    def ApplyVisitIncrements(self, path):
        for parent, action in path:
            parent.node.total_visits += 1
            parent.node.visit_count[action] += 1

    # ----------------------------------------------------------
    # Perform one root-level MCTS search
    # ----------------------------------------------------------
    def search(self, root_state:Table):
        
        self.nodes.clear()

        root = MCTSNode().from_table(root_state.clone()) # certainly not forced

        # initial expansion + noise
        self.expand(root)
        self.add_root_noise(root)

        for _ in range(self.num_simulations):

            node = root
            path = []

            # -------------------------
            # SELECTION
            # -------------------------
            while node.node.is_expanded and node.node.priors:

                mv = self.select_move(node)
                if mv is None:
                    break

                path.append((node, mv))

                # create child if missing
                if mv not in node.children:
                    child_state = node.state.clone() # table
                    child_state.apply_move(mv)

                    # forced move collapse
                    fm = heuristic_forced_move(child_state)
                    if isinstance(fm, dict) and 'domino' in fm: moves = fm['domino']
                    elif isinstance(fm, tuple): moves = [fm]
                    else: moves, _, _ = generate_root_moves_with_collapse(child_state)
                    while (len(moves) == 1): # walk again and again (can be optimalised)
                        child_state.apply_move(moves[0])
                        fm = heuristic_forced_move(child_state)
                        if isinstance(fm, dict) and 'domino' in fm: break
                        elif isinstance(fm, tuple): moves = [fm]
                        else: moves, _, _ = generate_root_moves_with_collapse(child_state)

                    # no forced now

                    # setup decorated DAP structure (tree with repeated nodes)
                    hash_ = child_state.__hash__()
                    realNew = False
                    if (self.nodes.__contains__(hash_)):
                        # already explored in tree as different node but similar state
                        others_inner_node = self.nodes[hash_]
                        child = MCTSNode().from_node(others_inner_node, child_state.FirstPlayer, child_state.score)
                        
                        # continue
                    else:
                        # new node
                        realNew = True
                        child = MCTSNode().from_table(child_state)
                        self.nodes[hash_] = child.node

                    

                    node.children[mv] = child
                    node = child
                        
                        
                    if (realNew): break
                else:
                    node = node.children[mv]

            # -------------------------
            # EXPAND / EVALUATE OR TERMINAL
            # -------------------------
            if not node.node.is_expanded: # new one
                #print(len(path)*'\t' + str(mv))

                self.ApplyVisitIncrements(path)
                self.expand(node)

                results = self.evaluate_leaf(node, path)
                if results:
                    # backpropagate all returned results
                    for (evaluated_node, val, eval_path) in results:
                        self.PropBack(val, evaluated_node, eval_path)

                continue # continue to next simulation because leaf evaluation happens asynchronously

            if node.state.game_over():

                v_net = 0.0       # consistent normalized terminal value
                self.ApplyVisitIncrements(path)
                self.PropBack(v_net, node, path)
                continue
            else: # wut?
                raise Exception("wut?")
                # non-terminal leaf already expanded earlier (rare), queue for eval
                results = self.evaluate_leaf(node, path)
                if results:
                    for (evaluated_node, val, eval_path) in results:
                        self.PropBack(val, evaluated_node, eval_path)
                continue

        # ------------------------------------------
        # PICK BEST ROOT MOVE
        # ------------------------------------------
        best_mv = None
        best_vis = -1
        best_avg = -1e9

        for mv in root.node.priors.keys():
            N = root.node.visit_count[mv]
            avg = (root.node.value_sum[mv]/N) if N > 0 else 0.0

            if N > best_vis or (N == best_vis and avg > best_avg):
                best_vis = N
                best_avg = avg
                best_mv = mv

        if best_mv is None:
            raise Exception()

        if(len(self.batcher.pending) > 0): # flush remaining pending evaluations
            #raise Exception()
            self.batcher.flush()
            #for (evaluated_node, val, eval_path) in self.batcher.flush(): self.PropBack(val, evaluated_node, eval_path)

        return best_mv, root.node

# -------------------------
# Self-play with heuristics + domino handling + MCTS
# -------------------------
def self_play_episode(mcts:MCTS, N:int, prefills:int, rng:random.Random):
    t = Table(N)
    # rng games first
    for _ in range(prefills):
        if t.game_over(): break
        mv = rng.choice(t.legal_moves()); t.apply_move(mv)


    root_was_first=[]
    root_encoded_states=[]
    root_scores = []
    root_remaining_boxes = []
    root_mcts_values = []
    while not t.game_over():
        root_encoded_states.append(encode_table(t))
        root_was_first.append(t.FirstPlayer)
        root_scores.append(t.score)
        root_remaining_boxes.append(t.remaining_boxes)

        forced = heuristic_forced_move(t)
        
        if isinstance(forced, tuple): t.apply_move(forced)
        else:
            mv, root = mcts.search(t)
            #print('best move:', mv)

            # Compute MCTS root value estimate (30% of teach target)
            N_total = max(1, root.total_visits)
            root_v = 0.0
            for mv2, N in root.visit_count.items(): root_v += root.value_sum[mv2]
            root_v /= N_total
            root_mcts_values.append(root_v)

            t.apply_move(mv)

        # forced is either a move tuple, or {'domino': [mvA,mvB]}
    
    final_score = t.score
    print("selfplay score:", t.score)
    return root_encoded_states, root_was_first, root_scores, root_mcts_values, root_remaining_boxes, final_score

# -------------------------
# Trainer (save/load)
# -------------------------
class Trainer:
    def __init__(self, board_size=4, mcts_num_sim=80, device='cpu', lr=2e-4, replay_capacity=50000):
        self.device = torch.device(device)
        self.board_size = board_size
        self.model = DotsValueNet(board_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.replay = ReplayBuffer(replay_capacity)

        self.mcts = MCTS(
            model=self.model,
            device=self.device,
            num_simulations=mcts_num_sim,  # default, overridden during train_iterations
        )

    def save(self, path):
        path = os.path.join(GetPath(), "models", path)
        torch.save({'model': self.model.state_dict(),
                    'optim': self.optimizer.state_dict()}, path)

    def load(self, path):
        path = os.path.join(GetPath(), "models", path)
        data = torch.load(path, map_location=self.device)
        self.model.load_state_dict(data['model'])
        if 'optim' in data:
            try:
                self.optimizer.load_state_dict(data['optim'])
            except Exception:
                pass
        print(f"Loaded model from {path}")

    def warmup_replay(self, episodes=50, seed=0):
        rng = random.Random(seed)
        print("Warmup replay generation with weak MCTS...")

        # temporary: very small MCTS simulations
        old_sims = self.mcts.num_simulations
        self.mcts.num_simulations = 8

        for ep in range(episodes):
            # large prefills to keep positions simple in warmup
            prefills = 10

            states, firsts, scores, mcts_vals, rems, final = \
                self_play_episode(self.mcts, self.board_size, prefills, rng)

            for s, fp, sc, vm, r in zip(states, firsts, scores, mcts_vals, rems):
                if not fp: vm = -vm
                target = 0.7*((final - sc)/r) #+ 0.3*vm # in warmup MCTS less relyable. (we will directly learn final scores)
                if not fp: target = -target
                self.replay.push(s, target)

        # restore simulation count
        self.mcts.num_simulations = old_sims

    def train_iterations(self, total_iters=1000, episodes_per_iter=4,
                         prefill_start=12, prefill_end=0,
                         batch_size=128, seed=42, log_every=25):

        rng = random.Random(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # 1) Warmup replay (weak MCTS)
        self.warmup_replay(episodes=60, seed=seed)

        # 2) Main training loop
        for it in range(1, total_iters+1):
            frac = it / total_iters
            # PREFILL CSÖKKEN
            prefills = int(prefill_start + (prefill_end - prefill_start) * frac)
            self.mcts.num_simulations = int(20 + frac * 60)
            for _ in range(episodes_per_iter):
                states, firsts, scores, mcts_vals, rems, final = \
                    self_play_episode(self.mcts, self.board_size, prefills, rng)

                # targets
                for s, fp, sc, vm, r in zip(states, firsts, scores, mcts_vals, rems):
                    if not fp: vm = -vm
                    target = 0.7 * ((final - sc) / r) + 0.3 * vm
                    if not fp: target = -target
                    self.replay.push(s, target)

            # TRAINING STEPS
            if len(self.replay) >= 64:
                for _ in range(4):
                    st, tg = self.replay.sample(batch_size)
                    stt = torch.from_numpy(st).float().to(self.device)
                    tgt = torch.from_numpy(tg).float().to(self.device)

                    self.model.train()
                    pred = self.model(stt)
                    loss = F.mse_loss(pred, tgt)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            if it % log_every == 0:
                print(f"Iter {it}/{total_iters}  | replay={len(self.replay)}  | prefills={prefills}  | sims={self.mcts.num_simulations}")

            if it % 100 == 0:
                self.save(f"dots_value_checkpoint_it{it}.pt")

        self.save("dots_value_final.pt")
        print("Training finished.")

# -------------------------
# Pygame UI updated for domino choices for human
# -------------------------


class PygameUI:
    def __init__(self, trainer:Trainer, board_size=4, mcts_sim=80, heuristic_first=True, heuristic_help=True):
        pygame.init()
        self.trainer = trainer
        #self.device = trainer.device
        #self.model = trainer.model
        self.board_size = board_size
        #self.mcts_sim = mcts_sim
        self.heuristic_first = heuristic_first
        self.heuristic_help = heuristic_help
        self.cell = 60
        self.margin = 20
        self.width = self.margin*2 + (self.cell * board_size)
        self.height = self.margin*2 + (self.cell * board_size) + 40
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Dots & Boxes - AI")
        self.font = pygame.font.SysFont(None, 24)

    def draw_board(self, table:Table, highlight_moves:List[Tuple[str,int,int]]=None):
        self.screen.fill((30,30,30))
        N = table.N
        cell = self.cell
        m = self.margin
        if highlight_moves is None: highlight_moves = []
        suicide_edges = getattr(self, "_suicide_edges", set())
        collapsed_edges = getattr(self, "_collapsed_edges", set())

        # draw filled boxes (color by owner)
        for by in range(N):
            for bx in range(N):
                owner = table.owner[by, bx]   # should be 1 or 2 or 0/unfilled
                if owner == 1:
                    color = (230, 230, 120)   # light yellow
                elif owner == 2:
                    color = (120, 230, 230)   # light cyan
                else:
                    continue  # not completed yet

                px = m + bx * cell
                py = m + by * cell
                pygame.draw.rect(self.screen, color, (px+3, py+3, cell-5, cell-5))

        # horizontal lines
        for y in range(N+1):
            for x in range(N):
                px = m + x*cell
                py = m + y*cell
                mv = ('h', x, y)
                if table.horizontal[y,x]==1:
                    pygame.draw.line(self.screen,(200,200,80),(px,py),(px+cell,py),6)
                else:
                    if (not self.heuristic_help):
                        pygame.draw.line(self.screen,(60,60,60),(px,py),(px+cell,py),4)
                    elif mv in highlight_moves:
                        pygame.draw.line(self.screen,(100,250,100),(px,py),(px+cell,py),6)
                    elif mv in suicide_edges:
                        pygame.draw.line(self.screen, (220,50,50), (px,py),(px+cell,py),6)  # red
                    elif mv in collapsed_edges:
                        pygame.draw.line(self.screen, (255,140,0), (px,py),(px+cell,py),6)  # orange
                    else:
                        pygame.draw.line(self.screen,(60,60,60),(px,py),(px+cell,py),4) # faint, clickable

        # vertical lines
        for y in range(N):
            for x in range(N+1):
                px = m + x*cell
                py = m + y*cell
                mv = ('v', x, y)
                if table.vertical[y,x]==1:
                    pygame.draw.line(self.screen,(80,200,200),(px,py),(px,py+cell),6)
                else:
                    if (not self.heuristic_help):
                        pygame.draw.line(self.screen,(60,60,60),(px,py),(px,py+cell),4)
                    elif mv in highlight_moves:
                        pygame.draw.line(self.screen,(100,250,100),(px,py),(px,py+cell),6)
                    elif mv in suicide_edges:
                        pygame.draw.line(self.screen, (220,50,50),(px,py),(px,py+cell),6)
                    elif mv in collapsed_edges:
                        pygame.draw.line(self.screen, (255,140,0),(px,py),(px,py+cell),6)
                    else:
                        pygame.draw.line(self.screen,(60,60,60),(px,py),(px,py+cell),4)

        # draw dots
        for y in range(N+1):
            for x in range(N+1):
                px = m + x*cell
                py = m + y*cell
                pygame.draw.circle(self.screen, (220,220,220), (px,py), 4)

        # scores
        s = f"Score (P1 - P2): {table.score}"
        t = self.font.render(s, True, (240,240,240))
        self.screen.blit(t, (10, self.height-45))
        p = f"FirstPlayer: {table.FirstPlayer}"
        t = self.font.render(p, True, (240,240,240))
        self.screen.blit(t, (10, self.height-25))
        pygame.display.flip()


    def play_human_vs_ai(self):
        table = Table(self.board_size, UI=True)
        running = True
        human_is_p1 = True
        clock = pygame.time.Clock()
        while running:
            # draw & event
            # compute highlights once per loop/turn
            # suicidal edges (red)

            # collapsed open components -> orange edges
            self._collapsed_edges = set()
            _, self._suicide_edges, self._collapsed_edges = generate_root_moves_with_collapse(table) # for color

            # compute forced/domino highlights as before
            forced = heuristic_forced_move(table)
            highlight = []
            if human_is_p1 == table.FirstPlayer:
                if isinstance(forced, dict) and 'domino' in forced: highlight = forced['domino']
                elif isinstance(forced, tuple):                     highlight = [forced]

            self.draw_board(table, highlight_moves=highlight)

            if table.game_over():
                print("Game over. Final score (P1 - P2) =", table.score)
                pygame.time.wait(2000); return

            for event in pygame.event.get():
                if event.type==pygame.QUIT:
                    pygame.quit(); sys.exit()
                if event.type==pygame.MOUSEBUTTONDOWN:
                    mx,my = event.pos
                    mv = self.mouse_to_move(mx,my,table)
                    if mv:
                        # if domino_mode, only allow those two
                        if self.heuristic_help and highlight and mv not in highlight: continue

                        # only allow human to move when it's their turn
                        if table.FirstPlayer == human_is_p1: table.apply_move(mv)

            # AI's turn
            if table.FirstPlayer != human_is_p1:
                forced = heuristic_forced_move(table)
                if isinstance(forced, tuple): 
                    table.apply_move(forced)
                else:
                    mv, _ = self.trainer.mcts.search(table)

                    print("enemy move:", mv)
                    table.apply_move(mv)
                        
            clock.tick(30)

    def play_ai_vs_ai(self, render=True, delay=0.2):
        t = Table(self.board_size, UI=True)
        if render: self.draw_board(t)
        while not t.game_over():

             for event in pygame.event.get():
                if event.type==pygame.MOUSEBUTTONDOWN:

                    forced = heuristic_forced_move(t)
                    if isinstance(forced, tuple): 
                        t.apply_move(forced)
                        if render: self.draw_board(t)
                        continue

                    mv, _ = self.trainer.mcts.search(t)
                    print(mv)
                    t.apply_move(mv)

                    if render: self.draw_board(t)
                    #if render: pygame.time.wait(int(delay*1000))
        print("AI vs AI finished. score:", t.score)
        if render: pygame.time.wait(2000)

    def play_human_vs_human(self):
        table = Table(self.board_size, UI=True)
        running = True
        clock = pygame.time.Clock()

        while running:

            # compute collapsed edges (orange)
            self._collapsed_edges = set()
            _, self._suicide_edges, self._collapsed_edges = generate_root_moves_with_collapse(table) # for color

            # forced moves (domino or single)
            forced = heuristic_forced_move(table)
            highlight = []

            if isinstance(forced, dict) and 'domino' in forced: highlight = forced['domino'][:] # highlight 2 moves 
            elif isinstance(forced, tuple):                     highlight = [forced]            # highlight 1 move

            # draw
            self.draw_board(table, highlight_moves=highlight)

            if table.game_over():
                print("Game over. Final score (P1 - P2) =", table.score)
                pygame.time.wait(2000)
                return

            # --- Event loop ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = event.pos
                    mv = self.mouse_to_move(mx, my, table)

                    if mv:
                        # require forced/domino mode
                        if self.heuristic_help and highlight and mv not in highlight: continue

                        table.apply_move(mv)

            clock.tick(30)


    def mouse_to_move(self,mx,my,table:Table):
        N=table.N; cell=self.cell; m=self.margin
        for y in range(N+1):
            for x in range(N):
                px = m + x*cell; py = m + y*cell
                rect = pygame.Rect(px-10, py-10, cell+20, 20)
                if rect.collidepoint(mx,my):
                    if table.horizontal[y,x]==0: return ('h',x,y)
        for y in range(N):
            for x in range(N+1):
                px = m + x*cell; py = m + y*cell
                rect = pygame.Rect(px-10, py-10, 20, cell+20)
                if rect.collidepoint(mx,my):
                    if table.vertical[y,x]==0: return ('v',x,y)
        return None

# -------------------------
# CLI & main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['play','train','selfplay','pvp'], help='Mode')
    parser.add_argument('--board', type=int, default=4)
    parser.add_argument('--guide', type=int, default=1) # help player with heuristics
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--load', default=None)
    parser.add_argument('--mcts', type=int, default=80)
    parser.add_argument('--iters', type=int, default=800)
    args = parser.parse_args()

    trainer = Trainer(board_size=args.board, mcts_num_sim=args.mcts, device=args.device)
    if args.load:
        trainer.load(args.load)
        #trainer.load('dots_value_checkpoint_it200.pt')

    #debug
    #trainer.train_iterations(total_iters=args.iters, episodes_per_iter=4, prefill_start=0, prefill_end=args.board*args.board, batch_size=128)

    #ui = PygameUI(trainer, board_size=args.board, mcts_sim=args.mcts, heuristic_help=(args.guide==1)) ; ui.play_human_vs_ai()

    #ui = PygameUI(trainer, board_size=args.board, mcts_sim=args.mcts) ;ui.play_ai_vs_ai(render=True)

    #ui = PygameUI(trainer, board_size=args.board, mcts_sim=args.mcts, heuristic_help=(args.guide==1)) ; ui.play_human_vs_human()

    if args.mode == 'train':
        trainer.train_iterations(total_iters=args.iters, episodes_per_iter=4,
                                 prefill_start=0, prefill_end=args.board*args.board, batch_size=128)

    if args.mode == 'play':
        ui = PygameUI(trainer, board_size=args.board, mcts_sim=args.mcts, heuristic_help=(args.guide==1))
        ui.play_human_vs_ai()
    elif args.mode == 'selfplay':
        ui = PygameUI(trainer, board_size=args.board, mcts_sim=args.mcts)
        ui.play_ai_vs_ai(render=True)
    elif args.mode == 'pvp':       # NEW MODE — Human vs Human
        ui = PygameUI(trainer, board_size=args.board, mcts_sim=args.mcts, heuristic_help=(args.guide==1))
        ui.play_human_vs_human()

if __name__ == '__main__':
    main()
