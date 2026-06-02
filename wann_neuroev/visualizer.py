import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import sys
import math

from Lab2.wann_neuroev.evolution import INPUT, OUTPUT, HIDDEN

class Visualizer:
    def __init__(self, window_size: tuple[int, int], fps=30, n_radius=18,
                 border_thickness=4, con_thickness=2):

        if not pygame.get_init():
            pygame.init()
        pygame.font.init()

        self.base_width = window_size[0]
        self.base_height = window_size[1]

        self.width = window_size[0]
        self.height = window_size[1]
        self.window = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.fps = fps
        self.n_radius = n_radius
        self.border_thickness = border_thickness
        self.con_thickness = con_thickness
        self.colors = {
            'input': (40, 102, 51),
            'output': (133, 44, 44),
            'hidden': (30, 30, 30),
            'background': (255, 255, 255),
            'connection': (90, 90, 90),
            'white': (255, 255, 255),
            'text': (30, 30, 30)
        }

        self.font = pygame.font.SysFont(None, 36)
        self.text_cache = {}

    def _get_text_surface(self, text, alpha, current_radius, current_border):
        cache_key = (text, alpha, current_radius)
        if cache_key in self.text_cache:
            return self.text_cache[cache_key]

        text_surf = self.font.render(text, True, self.colors['text'])

        max_width = (current_radius - current_border) * 1.7
        if text_surf.get_width() > max_width and max_width > 0:
            scale = max_width / text_surf.get_width()
            new_size = (max(1, int(text_surf.get_width() * scale)), max(1, int(text_surf.get_height() * scale)))
            text_surf = pygame.transform.smoothscale(text_surf, new_size)

        text_surf.set_alpha(alpha)
        self.text_cache[cache_key] = text_surf
        return text_surf

    def draw_net(self, individual, mouse_pos, column_size=3):
        scale = min(self.width / self.base_width, self.height / self.base_height)
        c_radius = max(5, int(self.n_radius * scale))
        c_border = max(1, int(self.border_thickness * scale))
        c_con = max(1, int(self.con_thickness * scale))

        input_ids = individual.get_node_ids(INPUT)
        output_ids = individual.get_node_ids(OUTPUT)
        hidden_ids = individual.get_node_ids(HIDDEN)

        hidden_cols = [hidden_ids[i:i + column_size] for i in range(0, len(hidden_ids), column_size)]
        all_columns = [input_ids] + hidden_cols + [output_ids]

        node_positions = {}
        total_cols = len(all_columns)
        x_step = self.width / (total_cols + 1)

        for col_idx, col_nodes in enumerate(all_columns):
            x = int((col_idx + 1) * x_step)
            y_step = self.height / (len(col_nodes) + 1)
            for node_idx, node_id in enumerate(col_nodes):
                y = int((node_idx + 1) * y_step)
                node_positions[node_id] = (x, y)

        hovered_node = None
        for node_id, pos in node_positions.items():
            dist = math.hypot(pos[0] - mouse_pos[0], pos[1] - mouse_pos[1])
            if dist <= c_radius:
                hovered_node = node_id
                break

        active_nodes = set()
        active_connections = set()

        if hovered_node is not None:
            active_nodes.add(hovered_node)
            for start_id, end_id in individual.connections:
                if start_id == hovered_node or end_id == hovered_node:
                    active_connections.add((start_id, end_id))
                    active_nodes.add(start_id)
                    active_nodes.add(end_id)
        else:
            active_nodes = set(node_positions.keys())
            active_connections = individual.connections

        surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        self.window.fill(self.colors['background'])

        for start_id, end_id in individual.connections:
            if start_id in node_positions and end_id in node_positions:
                is_active = (start_id, end_id) in active_connections
                alpha = 255 if is_active else 40

                color_with_alpha = (*self.colors['connection'][:3], alpha)

                pygame.draw.line(
                    surf,
                    color_with_alpha,
                    node_positions[start_id],
                    node_positions[end_id],
                    c_con
                )

        for node_id, pos in node_positions.items():
            node = individual.nodes[node_id]
            is_active = node_id in active_nodes
            alpha = 255 if is_active else 40

            if node.type == INPUT:
                base_color = self.colors['input']
            elif node.type == OUTPUT:
                base_color = self.colors['output']
            else:
                base_color = self.colors['hidden']

            color_with_alpha = (*base_color[:3], alpha)
            white_with_alpha = (*self.colors['white'][:3], alpha)

            pygame.draw.circle(surf, color_with_alpha, pos, c_radius)
            pygame.draw.circle(surf, white_with_alpha, pos, c_radius - c_border)

            text_surf = self._get_text_surface(node.activation, alpha, c_radius, c_border)
            text_rect = text_surf.get_rect(center=pos)
            surf.blit(text_surf, text_rect)

        self.window.blit(surf, (0, 0))
        pygame.display.flip()

    def draw(self, individual, column_size=3):
        running = True
        while running:
            mouse_pos = pygame.mouse.get_pos()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    self.width, self.height = event.w, event.h
                    self.window = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)

            self.draw_net(individual, mouse_pos, column_size)
            self.clock.tick(self.fps)

        pygame.quit()
