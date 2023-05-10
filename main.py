import pygame
import numpy as np
import scipy.linalg as sla

# Pygame initialization
pygame.init()
width, height = int(16 * 80), int(9 * 80)
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

# Plot parameters
ymin = -2
xmin = ymin * 16 / 9
xmax = abs(xmin)
ymax = abs(ymin)
frames = 400

# Calculate center offset for plot
center_x = width // 2
center_y = height // 2

# Wave packet parameters
NN = 2000
pwp = 40
sig = 0.15
Vmax = pwp**2 / 2 * 1.0

dx = (2 * xmax - 2 * xmin) / NN
xvec = np.arange(xmin * 2, xmax * 2, dx)
b = -1 / (2 * dx**2)

# Finite difference Laplacian
diag = -np.diagflat(2 * b * np.ones(NN))
offdiag = np.diagflat(b * np.ones(NN - 1), 1)
D2 = diag + offdiag + offdiag.transpose()
pot = np.zeros(NN)
for ii in range(NN):
    if int(0.495 * NN) < ii < int(0.505 * NN):
        pot[ii] = Vmax
H = D2 + np.diagflat(pot)

ev, evec = sla.eigh(H)
idx = ev.argsort()
ev, evec = ev[idx], evec[idx]

initwf = np.exp(-(xvec + 2)**2 / sig - 1j * pwp * (xvec + 2))  # initial Gaussian
init = ev[0]
cm = np.array([np.sum(initwf * evec[:, mm]) for mm in range(len(ev))])
cm = cm / sla.norm(cm)
psi = [cm[jj] * evec[:, jj] for jj in range(len(ev))]

# Animation loop
running = True
frame = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((0, 0, 0))

    # Calculate wave function for current frame
    tt2 = frame / frames
    Psi = psi[0] * np.exp(1j * ev[0] * tt2)
    for jj in range(1, len(psi)):
        Psi += psi[jj] * np.exp(1j * ev[jj] * tt2)

    # Plot the wave function
    points = np.column_stack((xvec, 5 * np.abs(Psi)**2))
    points = np.multiply(points, [width / (2 * xmax), height / (2 * ymax)])
    points = points.astype(int)
    points += [center_x, center_y]

    pygame.draw.lines(screen, (255, 0, 0), False, points, 2)

    # Draw potential barrier
    barrier_points = np.column_stack((xvec, pot / abs(b) * 0.5))
    barrier_points = np.multiply(barrier_points, [width / (2 * xmax), height / (2 * ymax)])
    barrier_points = barrier_points.astype(int)
    barrier_points += [center_x, center_y]
    pygame.draw.polygon(screen, (200, 200, 200), barrier_points, 1)

    pygame.display.flip()
    frame += 1

    if frame >= frames:
        running = False

    clock.tick(60)

pygame.quit