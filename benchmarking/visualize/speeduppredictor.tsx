import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';

interface PredictionFeatures {
  ndim: number;
  total_size: number;
  max_dim: number;
  min_dim: number;
  mean_dim: number;
  std_dim: number;
  num_axes_reduced: number;
  max_reduced_dim: number;
  min_reduced_dim: number;
  reduced_size: number;
  kept_size: number;
}

const SpeedupPredictor: React.FC = () => {
  const [shape, setShape] = useState<number[]>([1000, 1000, 100]);
  const [axis, setAxis] = useState<number[]>([0, 1]);
  const [prediction, setPrediction] = useState<number | null>(null);

  const generateFeatures = (shape: number[], axis: number[]): PredictionFeatures => {
    return {
      ndim: shape.length,
      total_size: shape.reduce((a, b) => a * b, 1),
      max_dim: Math.max(...shape),
      min_dim: Math.min(...shape),
      mean_dim: shape.reduce((a, b) => a + b, 0) / shape.length,
      std_dim: Math.sqrt(shape.reduce((a, b) => a + Math.pow(b - shape.reduce((a, b) => a + b, 0) / shape.length, 2), 0) / shape.length),
      num_axes_reduced: axis.length,
      max_reduced_dim: Math.max(...axis.map(i => shape[i])),
      min_reduced_dim: Math.min(...axis.map(i => shape[i])),
      reduced_size: axis.map(i => shape[i]).reduce((a, b) => a * b, 1),
      kept_size: shape.filter((_, i) => !axis.includes(i)).reduce((a, b) => a * b, 1)
    };
  };

  const handleShapeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newShape = e.target.value.split(',').map(x => parseInt(x.trim())).filter(x => !isNaN(x));
    setShape(newShape);
  };

  const handleAxisChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newAxis = e.target.value.split(',').map(x => parseInt(x.trim())).filter(x => !isNaN(x));
    setAxis(newAxis);
  };

  const handlePredict = () => {
    const features = generateFeatures(shape, axis);
    // This is where you'd make an API call to your model
    // For now, using a simple heuristic
    const predicted = (features.total_size / 1e6) * (features.num_axes_reduced / features.ndim) * 2;
    setPrediction(predicted);
  };

  return (
    <Card className="w-full max-w-4xl">
      <CardHeader>
        <CardTitle>Numba Speedup Predictor</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-1">Shape (comma-separated)</label>
              <input
                type="text"
                className="w-full p-2 border rounded"
                value={shape.join(', ')}
                onChange={handleShapeChange}
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">Reduction Axes (comma-separated)</label>
              <input
                type="text"
                className="w-full p-2 border rounded"
                value={axis.join(', ')}
                onChange={handleAxisChange}
              />
            </div>
          </div>
          
          <button
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
            onClick={handlePredict}
          >
            Predict Speedup
          </button>
          
          {prediction !== null && (
            <div className="mt-4 p-4 bg-gray-100 rounded">
              <h3 className="text-lg font-semibold">Predicted Speedup</h3>
              <p className="text-2xl font-bold">{prediction.toFixed(2)}x</p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default SpeedupPredictor;