# SEA Alignment Viewer - Frontend

React TypeScript GUI for visualizing image alignment results from the SEA pipeline.

## Features

### 3-Step Visualization Flow

1. **Before/After Comparison** - Side-by-side view of original vs aligned images
2. **Movement Quantification** - Detailed table showing alignment corrections per frame
3. **Final Results** - Browse and download all output files

## Quick Start

### Prerequisites

- Node.js 18+ and npm
- Python backend running (see Backend Setup below)

### Installation

```bash
cd frontend
npm install
```

### Development Mode (with Mock Data)

```bash
npm run dev
```

Then open http://localhost:3000 in your browser. By default, the app uses mock data so you can develop without a backend.

### Production Mode (with Real Backend)

1. Start the Flask backend (from project root):
```bash
conda activate SEA
python api_server.py
```

2. In another terminal, start the frontend:
```bash
cd frontend
npm run dev
```

3. In the browser, uncheck "Use Mock Data" to connect to the real backend.

## Project Structure

```
frontend/
├── src/
│   ├── components/          # React components
│   │   ├── AlignmentViewer.tsx       # Main viewer
│   │   ├── BeforeAfterView.tsx       # Step 1
│   │   ├── MovementQuantification.tsx # Step 2
│   │   ├── ResultsVisualization.tsx  # Step 3
│   │   └── FrameSelector.tsx         # Frame selection
│   ├── types/               # TypeScript types
│   │   └── alignment.ts
│   ├── mocks/               # Mock data for development
│   │   └── mockData.ts
│   ├── styles/              # CSS files
│   └── App.tsx              # Root component
├── public/                  # Static assets
├── package.json
├── tsconfig.json
└── vite.config.ts
```

## Available Scripts

- `npm run dev` - Start development server (port 3000)
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

## Configuration

### API Proxy

The development server proxies API requests to the Flask backend. Configure in `vite.config.ts`:

```typescript
server: {
  proxy: {
    '/api': {
      target: 'http://localhost:5000',  // Flask backend URL
      changeOrigin: true,
    },
  },
}
```

## Data Contract

### Alignment Result API (`/api/alignment/<sample_name>`)

```typescript
{
  sampleName: string;
  frames: AlignmentFrame[];
  finalResults: ResultItem[];
  metadata: {
    anchorChannel: string;
    totalChannels: number;
  };
}
```

### AlignmentFrame

```typescript
{
  frameId: string;
  channelName: string;
  beforeImageUrl: string;
  afterImageUrl: string;
  movement: {
    dx: number;           // Horizontal movement (pixels)
    dy: number;           // Vertical movement (pixels)
    magnitude: number;    // Total movement magnitude
    transformType: 'affine' | 'tps' | 'identity';
    residualError?: number;
    numMatches?: number;
  };
}
```

## Customization

### Styling

All styles are in `src/styles/`. Each component has its own CSS file:
- `AlignmentViewer.css`
- `BeforeAfterView.css`
- `MovementQuantification.css`
- `ResultsVisualization.css`
- `FrameSelector.css`

### Mock Data

Edit `src/mocks/mockData.ts` to customize mock data for development.

## Troubleshooting

### CORS Errors

If you see CORS errors, ensure the Flask backend has `flask-cors` installed:
```bash
pip install flask-cors
```

### Images Not Loading

- Check that the Flask backend is running
- Verify the sample exists in `data/output/<sample_name>/`
- Check browser console for specific error messages

### TypeScript Errors

Run the TypeScript compiler to check for errors:
```bash
npm run build
```

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)

## License

[Same as parent project]


