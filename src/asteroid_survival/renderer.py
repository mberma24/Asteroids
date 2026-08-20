from __future__ import annotations

import math
import random
from collections import deque

from .state import WorldSnapshot


BG = (7, 10, 20)
WHITE = (230, 236, 246)
CYAN = (88, 220, 245)
ORANGE = (246, 155, 68)
RED = (241, 75, 75)


class Renderer:
    def __init__(self, pygame, width: int, height: int, *, trails: bool = False):
        self.pg = pygame
        # Trails and per-asteroid labels are for inspecting trajectory shapes. A shape is
        # very hard to judge from a moving dot; the path it has already flown is the thing
        # worth looking at.
        self.trails = trails
        self._trail: dict[int, deque] = {}
        self.logical_size = (width, height)
        desktop = pygame.display.get_desktop_sizes()[0]
        side = max(480, min(desktop) - 64)
        self.screen = pygame.display.set_mode((side, side), pygame.RESIZABLE)
        self.logical = pygame.Surface(self.logical_size)
        pygame.display.set_caption("Asteroid Survival")
        self.font = pygame.font.Font(None, 27)
        self.small = pygame.font.Font(None, 20)
        self.big = pygame.font.Font(None, 58)
        rng = random.Random(19)
        self.stars = [(rng.randrange(width), rng.randrange(height), rng.choice((1, 1, 1, 2))) for _ in range(150)]
        self._died_at: dict[str, float] = {}
        self._last_step = 0

    def _draw_trails(self, surface, state: WorldSnapshot) -> None:
        """Fading breadcrumbs behind each asteroid, so the flown path is visible.

        Drawn as separate dots rather than a polyline: the arena wraps, so joining points
        would streak a line straight across the screen every time one crosses an edge.
        """
        if state.step < self._last_step:
            self._trail.clear()
        live = {asteroid.id for asteroid in state.asteroids}
        for gone in [key for key in self._trail if key not in live]:
            del self._trail[gone]
        for asteroid in state.asteroids:
            path = self._trail.setdefault(asteroid.id, deque(maxlen=110))
            path.append((asteroid.x, asteroid.y))
            for index, (x, y) in enumerate(path):
                fade = (index + 1) / len(path)
                shade = (int(40 + 150 * fade), int(30 + 90 * fade), int(20 + 40 * fade))
                self.pg.draw.circle(surface, shade, (int(x), int(y)), 2)

    def resize(self, width: int, height: int) -> None:
        side = max(480, min(width, height))
        self.screen = self.pg.display.set_mode((side, side), self.pg.RESIZABLE)

    def _update_scores(self, state: WorldSnapshot) -> None:
        """Freeze each ship's survival time as it dies, so the result stays on screen."""
        if state.step < self._last_step:  # a restart rewinds the clock
            self._died_at.clear()
        self._last_step = state.step
        for ship in state.ships:
            if not ship.alive and ship.id not in self._died_at:
                self._died_at[ship.id] = state.elapsed

    def _draw_scoreboard(self, surface, state: WorldSnapshot, colors, top: int = 82) -> None:
        entries = []
        for index, ship in enumerate(state.ships):
            survived = self._died_at.get(ship.id, state.elapsed)
            entries.append((survived, ship.alive, ship.id, colors[index % len(colors)]))
        # Longest survivor first, with anyone still flying ranked above the dead.
        entries.sort(key=lambda entry: (entry[1], entry[0]), reverse=True)
        surface.blit(self.font.render("SURVIVAL", True, (130, 143, 169)), (18, top))
        for row, (survived, alive, ship_id, color) in enumerate(entries):
            status = f"{survived:5.1f}s" + ("" if alive else "  out")
            text = f"{row + 1}. {ship_id:<10}{status}"
            image = self.font.render(text, True, color if alive else (120, 128, 148))
            surface.blit(image, (18, top + 26 + row * 24))

    def draw(self, state: WorldSnapshot, paused: bool = False) -> None:
        pg, surface = self.pg, self.logical
        self._update_scores(state)
        surface.fill(BG)
        for x, y, r in self.stars:
            pg.draw.circle(surface, (65, 74, 99), (x, y), r)
        if state.objective.enabled:
            center = (int(state.objective.x), int(state.objective.y))
            pg.draw.circle(surface, (30, 93, 111), center, int(state.objective.radius + 8), 2)
            pg.draw.circle(surface, CYAN, center, int(state.objective.radius), 3)
        if self.trails:
            self._draw_trails(surface, state)
        for asteroid in state.asteroids:
            points = []
            for i in range(10):
                angle = i * math.tau / 10
                radius = asteroid.radius * (0.82 + 0.13 * math.sin(asteroid.id * 3.1 + i * 2.4))
                points.append((asteroid.x + math.cos(angle) * radius, asteroid.y + math.sin(angle) * radius))
            pg.draw.polygon(surface, ORANGE, points, 2)
        if self.trails:
            for asteroid in state.asteroids:
                label = self.small.render(asteroid.pattern, True, (255, 214, 150))
                surface.blit(label, label.get_rect(
                    center=(asteroid.x, asteroid.y - asteroid.radius - 11)))
        for projectile in state.projectiles:
            pg.draw.circle(surface, WHITE, (int(projectile.x), int(projectile.y)), int(projectile.radius))
        colors = (CYAN, (176, 117, 255), (101, 223, 137), (255, 219, 100))
        for i, ship in enumerate(state.ships):
            if not ship.alive:
                continue
            direction = (math.cos(ship.angle), math.sin(ship.angle))
            side = (-direction[1], direction[0])
            nose = (ship.x + direction[0] * ship.radius * 1.35, ship.y + direction[1] * ship.radius * 1.35)
            back_l = (ship.x - direction[0] * ship.radius + side[0] * ship.radius * .75,
                      ship.y - direction[1] * ship.radius + side[1] * ship.radius * .75)
            back_r = (ship.x - direction[0] * ship.radius - side[0] * ship.radius * .75,
                      ship.y - direction[1] * ship.radius - side[1] * ship.radius * .75)
            pg.draw.polygon(surface, colors[i % len(colors)], (nose, back_l, back_r), 2)
            label = self.font.render(ship.id, True, colors[i % len(colors)])
            surface.blit(label, label.get_rect(center=(ship.x, ship.y - ship.radius - 14)))
        alive = sum(s.alive for s in state.ships)
        wave = f"WAVE {state.wave}    " if state.wave else ""
        hud = (f"{wave}TIME {state.elapsed:7.1f}s    SHIPS {alive}/{len(state.ships)}"
               f"    ASTEROIDS {len(state.asteroids)}")
        if state.objective.enabled:
            hud += f"    OBJECT HP {max(0, state.objective.health)}"
        surface.blit(self.font.render(hud, True, WHITE), (18, 16))
        surface.blit(self.font.render("P pause  |  R restart  |  Esc quit", True, (130, 143, 169)), (18, 46))
        if state.difficulty is not None:
            # Endless runs get harder invisibly; show the knobs that are actually moving.
            d = state.difficulty
            tier = "" if d.tier is None else f"TIER {d.tier}    "
            ramp = (f"{tier}SPEED {d.min_speed:.0f}-{d.max_speed:.0f}    EVERY {d.spawn_interval:.2f}s"
                    f"    CAP {d.active_cap}    SWING {d.amplitude_max:.0f}"
                    f"    SPREAD {d.spawn_spread:.0f}\u00b0")
            surface.blit(self.font.render(ramp, True, ORANGE), (18, 76))
        if len(state.ships) > 1:
            self._draw_scoreboard(
                surface, state, colors, top=112 if state.difficulty is not None else 82)
        if paused or state.terminated or state.truncated:
            label = "PAUSED" if paused else f"RUN ENDED: {state.terminal_reason.value.replace('_', ' ').upper()}"
            image = self.big.render(label, True, RED if not paused else WHITE)
            surface.blit(image, image.get_rect(center=(state.width / 2, state.height / 2 - 45)))
            if not paused:
                msg = self.font.render("Press R to restart", True, WHITE)
                surface.blit(msg, msg.get_rect(center=(state.width / 2, state.height / 2 + 15)))
        if surface.get_size() == self.screen.get_size():
            self.screen.blit(surface, (0, 0))
        else:
            self.screen.blit(pg.transform.smoothscale(surface, self.screen.get_size()), (0, 0))
        pg.display.flip()
