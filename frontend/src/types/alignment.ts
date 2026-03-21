/**
 * TypeScript type definitions for SEA alignment visualization
 */

export interface AlignmentFrame {
  frameId: string;
  channelName: string;
  beforeImageUrl: string;
  afterImageUrl: string;
  movement: MovementData;
}

export interface MovementData {
  dx: number;  // Horizontal movement in pixels
  dy: number;  // Vertical movement in pixels
  magnitude: number;  // Total movement magnitude
  transformType: 'affine' | 'tps' | 'identity';
  residualError?: number;
  numMatches?: number;
}

export interface ResultItem {
  type: 'overlay' | 'csv' | 'plot' | 'label' | 'registered';
  name: string;
  url: string;
  description?: string;
}

export interface ChannelColocalization {
  channel: string;
  totalDetections: number;
  colocalizedCount: number;
  colocalizedPercent: number;
}

export interface ColocalizationStats {
  totalDetections: number;
  totalColocalized: number;
  colocalizationRate: number;
  channels: ChannelColocalization[];
}

export interface ComboEntry {
  combo: string[];  // Array of channel names, e.g., ["ch1", "ch3"]
  count: number;    // Number of objects with this combination
  rate: number;     // Fraction (0.0-1.0)
  percentage: number;  // Percentage (0-100)
}

export interface ComboAnalysis {
  image_id?: string;
  channels_present: string[];  // Normalized channel IDs (ch1, ch2, etc.)
  threshold_px: number;
  total_objects_grouped?: number;  // Optional - removed from display
  combo_table: ComboEntry[];  // Each entry.combo contains normalized channel IDs
}

export interface AlignmentResult {
  sampleName: string;
  frames: AlignmentFrame[];
  finalResults: ResultItem[];
  colocalization?: ColocalizationStats;
  comboAnalysis?: ComboAnalysis;  // NEW: Multi-channel combination analysis
  metadata?: {
    anchorChannel: string;
    totalChannels: number;
    processingTime?: number;
  };
}

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

