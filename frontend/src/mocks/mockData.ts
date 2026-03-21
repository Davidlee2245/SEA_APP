/**
 * Mock data for development and testing
 */

import { AlignmentResult, AlignmentFrame, ResultItem } from '../types/alignment';

/**
 * Generate mock alignment data for a sample
 */
export const getMockAlignmentData = (sampleName: string): AlignmentResult => {
  const channels = ['ch2-CD63', 'ch3-CD81', 'ch4-Syntenin', 'ch5-Marker'];
  
  const frames: AlignmentFrame[] = channels.map((channel, index) => {
    // Simulate varying degrees of movement
    const dx = (Math.random() - 0.5) * 10;
    const dy = (Math.random() - 0.5) * 10;
    const magnitude = Math.sqrt(dx * dx + dy * dy);
    
    return {
      frameId: `frame_${index + 1}`,
      channelName: channel,
      // Use placeholder images or actual paths if available
      beforeImageUrl: `/api/images/${sampleName}/preprocessed/${channel}_preprocessed.tif`,
      afterImageUrl: `/api/images/${sampleName}/registered/${channel}_registered.tif`,
      movement: {
        dx,
        dy,
        magnitude,
        transformType: magnitude > 5 ? 'tps' : magnitude > 1 ? 'affine' : 'identity',
        residualError: Math.random() * 2,
        numMatches: Math.floor(Math.random() * 500) + 100,
      },
    };
  });

  const finalResults: ResultItem[] = [
    {
      type: 'overlay',
      name: `${sampleName}_overlay.png`,
      url: `/api/results/${sampleName}/${sampleName}_overlay.png`,
      description: 'RGB overlay of all registered channels',
    },
    {
      type: 'csv',
      name: `${sampleName}_detections.csv`,
      url: `/api/results/${sampleName}/${sampleName}_detections.csv`,
      description: 'Quantification data with coordinates and intensities',
    },
    {
      type: 'plot',
      name: `${sampleName}_plots.png`,
      url: `/api/results/${sampleName}/${sampleName}_plots.png`,
      description: 'Statistical analysis plots',
    },
    ...channels.map((channel) => ({
      type: 'registered' as const,
      name: `${channel}_registered.tif`,
      url: `/api/results/${sampleName}/registered/${channel}_registered.tif`,
      description: `Registered image for ${channel}`,
    })),
    ...channels.map((channel) => ({
      type: 'label' as const,
      name: `${channel}_labels.tif`,
      url: `/api/results/${sampleName}/labels/${channel}_labels.tif`,
      description: `Detection labels for ${channel}`,
    })),
  ];

  return {
    sampleName,
    frames,
    finalResults,
    metadata: {
      anchorChannel: channels[0],
      totalChannels: channels.length,
      processingTime: Math.random() * 120 + 30,
    },
  };
};

/**
 * Mock API response
 */
export const mockApiResponse = <T>(data: T, delay: number = 500): Promise<T> => {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve(data);
    }, delay);
  });
};


