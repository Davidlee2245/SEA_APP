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
  type: 'overlay' | 'csv' | 'plot' | 'label' | 'registered' | 'json' | 'npz' | 'file';
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
  referenceColocalization?: ReferenceColocalizationRow[];
  exosomeColocalization?: ExosomeColocalizationResult;
  channels?: ExosomeChannelInfo[];
  metadata?: {
    anchorChannel: string;
    totalChannels: number;
    processingTime?: number;
    exosomeChannels?: string[];
    channelMarkers?: Record<string, string>;
    searchedExosomePath?: string;
  };
}

export interface ExosomeChannelInfo {
  id: string;
  name: string;
  biomarker?: string;
  has_results: boolean;
  has_masks: boolean;
  has_overlay: boolean;
  has_run: boolean;
  included: boolean;
  skip_reason?: string | null;
}

export interface ReferenceColocalizationRow {
  reference_object_id: number;
  reference_channel: string;
  reference_centroid_x: number;
  reference_centroid_y: number;
  reference_area: number;
  reference_perimeter: number;
  reference_circularity: number;
  biomarker_combination_label: string;
  total_positive_marker_count: number;
  overall_status: 'Positive' | 'Negative';
  positive_channels: string[];
  positive_biomarkers: string[];
  [key: string]: any;
}

export interface ExosomeColocalizationSummary {
  total_reference_objects: number;
  marker_positive_reference_objects: number;
  marker_negative_reference_objects: number;
  overall_positive_rate: number;
  total_matched_marker_objects: number;
  analysis_mode: 'overlap' | 'nearest_centroid';
  distance_threshold: number;
  reference_channel: string;
  marker_channels: string[];
}

export interface ExosomeColocalizationChannelSummary {
  channel: string;
  biomarker: string;
  total_marker_objects: number;
  positive_reference_objects: number;
  positive_rate: number;
  avg_marker_count_per_positive_reference: number;
  median_nearest_distance?: number | null;
}

export interface ExosomeColocalizationCombination {
  combination: string;
  count: number;
  rate: number;
  percentage: number;
}

export interface ExosomeColocalizationResult {
  reference_table: ReferenceColocalizationRow[];
  summary: ExosomeColocalizationSummary;
  channel_summary: ExosomeColocalizationChannelSummary[];
  combination_summary: ExosomeColocalizationCombination[];
  overlay?: {
    positive_reference_ids: number[];
    reference_channel: string;
  };
}

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

