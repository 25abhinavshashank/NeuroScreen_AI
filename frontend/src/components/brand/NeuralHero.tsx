const NODES = [
  { x: 60, y: 120, r: 3 },
  { x: 140, y: 60, r: 2.5 },
  { x: 210, y: 140, r: 4 },
  { x: 160, y: 220, r: 2.5 },
  { x: 280, y: 70, r: 3 },
  { x: 320, y: 180, r: 5 },
  { x: 400, y: 110, r: 3 },
  { x: 380, y: 230, r: 2.5 },
  { x: 460, y: 190, r: 3.5 },
  { x: 500, y: 90, r: 2.5 },
  { x: 90, y: 220, r: 2.5 },
  { x: 240, y: 260, r: 2.5 },
  { x: 440, y: 40, r: 2.5 },
  { x: 20, y: 60, r: 2 },
];

const EDGES: [number, number][] = [
  [0, 1],
  [1, 2],
  [2, 3],
  [0, 3],
  [2, 4],
  [4, 5],
  [5, 6],
  [6, 8],
  [8, 9],
  [5, 7],
  [3, 7],
  [1, 10],
  [3, 11],
  [7, 11],
  [6, 12],
  [9, 12],
  [10, 0],
  [1, 13],
];

export function NeuralHero({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 520 280"
      fill="none"
      className={className}
      aria-hidden="true"
      preserveAspectRatio="xMidYMid slice"
    >
      <defs>
        <linearGradient id="hero-line" x1="0" y1="0" x2="520" y2="280" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="hsl(var(--primary))" stopOpacity="0.55" />
          <stop offset="1" stopColor="hsl(var(--brand-accent))" stopOpacity="0.15" />
        </linearGradient>
        <radialGradient id="hero-node" cx="0.5" cy="0.5" r="0.5">
          <stop offset="0" stopColor="hsl(var(--brand-accent))" />
          <stop offset="1" stopColor="hsl(var(--primary))" />
        </radialGradient>
      </defs>

      {EDGES.map(([a, b], i) => (
        <line
          key={i}
          x1={NODES[a].x}
          y1={NODES[a].y}
          x2={NODES[b].x}
          y2={NODES[b].y}
          stroke="url(#hero-line)"
          strokeWidth="1"
        />
      ))}

      {NODES.map((n, i) => (
        <circle
          key={i}
          cx={n.x}
          cy={n.y}
          r={n.r}
          fill="url(#hero-node)"
          className="animate-pulse-node"
          style={{ animationDelay: `${(i % 6) * 0.5}s` }}
        />
      ))}
    </svg>
  );
}
