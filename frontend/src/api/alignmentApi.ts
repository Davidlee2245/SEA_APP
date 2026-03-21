/**
 * API client for alignment data
 */

import { AlignmentResult, ApiResponse } from '../types/alignment';
import { getApiBase } from '../lib/apiBase';

export class AlignmentApi {
  /**
   * Fetch alignment data for a sample
   */
  static async getAlignment(sampleName: string): Promise<AlignmentResult> {
    const response = await fetch(`${getApiBase()}/api/alignment/${sampleName}`);
    
    if (!response.ok) {
      throw new Error(`Failed to fetch alignment: ${response.statusText}`);
    }
    
    const data: ApiResponse<AlignmentResult> = await response.json();
    
    if (!data.success || !data.data) {
      throw new Error(data.error || 'Failed to load alignment data');
    }
    
    return data.data;
  }

  /**
   * List all available samples
   */
  static async listSamples(): Promise<string[]> {
    const response = await fetch(`${getApiBase()}/api/samples`);
    
    if (!response.ok) {
      throw new Error(`Failed to fetch samples: ${response.statusText}`);
    }
    
    const data: ApiResponse<string[]> = await response.json();
    
    if (!data.success || !data.data) {
      throw new Error(data.error || 'Failed to load samples');
    }
    
    return data.data;
  }

  /**
   * Health check
   */
  static async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${getApiBase()}/api/health`);
      return response.ok;
    } catch {
      return false;
    }
  }
}


