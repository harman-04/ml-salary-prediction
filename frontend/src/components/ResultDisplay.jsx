// src/components/ResultDisplay.jsx
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { motion, useMotionValue, useTransform, animate } from 'framer-motion';
import { useEffect } from 'react';
import { CheckCircle, XCircle, DollarSign, BarChart2, TrendingUp } from 'lucide-react';

// Import Recharts components
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ReferenceLine, Label
} from 'recharts';

const AnimatedCounter = ({ toValue }) => {
  const count = useMotionValue(0);
  const rounded = useTransform(count, latest => Math.round(latest));

  useEffect(() => {
    const controls = animate(count, toValue, { duration: 1.5, ease: "easeOut" });
    return controls.stop;
  }, [toValue, count]);

  return <motion.span>{rounded}</motion.span>;
};

// Component for Feature Importance Chart
const FeatureImportanceChart = ({ data }) => {
  if (!data || data.length === 0) {
    return (
      <CardDescription className="text-center text-gray-500 dark:text-gray-400 mt-4">
        No feature importance data available.
      </CardDescription>
    );
  }

  // Sort data by importance in descending order
  const sortedData = [...data].sort((a, b) => b.importance - a.importance);
  // Take top N features for clarity
  const topFeatures = sortedData.slice(0, Math.min(sortedData.length, 10));

  return (
    <Card className="w-full mt-6 shadow-lg border-blue-200 dark:border-blue-700">
      <CardHeader className="flex flex-row items-center space-x-2 pb-2">
        <BarChart2 className="w-5 h-5 text-blue-600 dark:text-blue-400" />
        <CardTitle className="text-lg font-semibold text-blue-800 dark:text-blue-200">Key Influencing Factors</CardTitle>
      </CardHeader>
      <CardContent className="h-72 lg:h-96 p-4">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={topFeatures}
            layout="vertical"
            // FIX 1: Increased left margin to give Y-axis labels more space
            margin={{ top: 5, right: 30, left: 30, bottom: 5 }} 
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
            <XAxis type="number" stroke="#666" />
            <YAxis
              type="category"
              dataKey="feature"
              // FIX 2: Increased width to match the new margin
              width={200}
              tickFormatter={(value) => {
                // FIX 3: More aggressive cleanup for better readability
                if (typeof value === 'string') {
                  return value
                    .replace(/_/g, ' ')
                    .replace(/HaveWorkedWith/g, '')
                    .replace(/num poly|cat ohe|num scale/gi, '') // Remove technical prefixes
                    .trim();
                }
                return String(value); // Convert to string if not already for display
              }}
              stroke="#666"
              fontSize={12}
            />
            <Tooltip
              cursor={{ fill: 'rgba(0,0,0,0.05)' }}
              contentStyle={{ backgroundColor: '#fff', border: '1px solid #ddd', borderRadius: '4px' }}
              labelStyle={{ color: '#333' }}
              formatter={(value) => [`Importance: ${value.toFixed(2)}`]}
            />
            <Legend wrapperStyle={{ paddingTop: '10px' }} />
            {/* FIX 4: Removed the redundant LabelList that was causing overlapping text */}
            <Bar dataKey="importance" fill="#4a90e2" name="Feature Importance" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

// Component for Salary Distribution Chart
const SalaryDistributionChart = ({ data, predictedSalary }) => {
  if (!data || data.length === 0) {
    return (
      <CardDescription className="text-center text-gray-500 dark:text-gray-400 mt-4">
        No salary distribution data available.
      </CardDescription>
    );
  }
  
  const minSalary = Math.min(...data, predictedSalary || Infinity);
  const maxSalary = Math.max(...data, predictedSalary || -Infinity);
  const numBins = 10;
  const binWidth = (maxSalary - minSalary) / numBins;

  const binnedData = Array.from({ length: numBins }).map((_, i) => {
    const lowerBound = minSalary + i * binWidth;
    const upperBound = lowerBound + binWidth;
    const count = data.filter(s => s >= lowerBound && s < upperBound).length;
    return {
      name: `${Math.round(lowerBound / 1000)}k - ${Math.round(upperBound / 1000)}k`,
      count: count,
      midpoint: (lowerBound + upperBound) / 2
    };
  });

  const predictedBinIndex = binnedData.findIndex(bin =>
    predictedSalary >= (bin.midpoint - binWidth / 2) && predictedSalary <= (bin.midpoint + binWidth / 2)
  );

  return (
    <Card className="w-full mt-6 shadow-lg border-green-200 dark:border-green-700">
      <CardHeader className="flex flex-row items-center space-x-2 pb-2">
        <TrendingUp className="w-5 h-5 text-green-600 dark:text-green-400" />
        <CardTitle className="text-lg font-semibold text-green-800 dark:text-green-200">Salary Landscape</CardTitle>
      </CardHeader>
      <CardContent className="h-72 lg:h-96 p-4">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={binnedData}
            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
            <XAxis dataKey="name" stroke="#666" fontSize={12} />
            <YAxis stroke="#666" />
            <Tooltip
              cursor={{ fill: 'rgba(0,0,0,0.05)' }}
              contentStyle={{ backgroundColor: '#fff', border: '1px solid #ddd', borderRadius: '4px' }}
              labelStyle={{ color: '#333' }}
              formatter={(value) => `${value} respondents`}
            />
            <Legend wrapperStyle={{ paddingTop: '10px' }} />
            <Bar dataKey="count" fill="#6bc0a6" name="Respondents" radius={[4, 4, 0, 0]} />
            {predictedSalary !== null && predictedBinIndex !== -1 && (
              <ReferenceLine
                x={binnedData[predictedBinIndex]?.name}
                stroke="red"
                strokeDasharray="5 5"
              >
                <Label value={`Your Prediction: $${predictedSalary.toLocaleString()}`} position="top" fill="red" fontSize={14} offset={10} />
              </ReferenceLine>
            )}
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};


export const ResultDisplay = ({ prediction, error, onRestart, featureImportances, salaryDistribution }) => {
  return (
    <Card className="w-full text-center border-2 border-dashed border-gray-300 dark:border-gray-700 overflow-auto max-h-[calc(100vh-100px)]">
      <CardHeader className="bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-t-lg py-6">
        <CardTitle className="text-3xl font-extrabold flex items-center justify-center gap-3">
          <DollarSign className="w-8 h-8" />
          Salary Prediction Result
        </CardTitle>
        <CardDescription className="text-blue-100 mt-2">
          Insights tailored to your profile.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col items-center justify-center space-y-8 p-8 bg-white dark:bg-gray-800 rounded-b-lg">
        {error ? (
          <>
            <XCircle className="w-20 h-20 text-red-500 animate-bounce" />
            <p className="text-xl font-bold text-red-600 dark:text-red-400">Prediction Failed!</p>
            <CardDescription className="text-red-500/80 dark:text-red-300 max-w-md">
              {error}. Please try again or check your input values.
            </CardDescription>
          </>
        ) : (
          <>
            <motion.div
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ type: "spring", stiffness: 200, damping: 20, delay: 0.2 }}
              className="flex flex-col items-center justify-center space-y-4"
            >
              <CheckCircle className="w-20 h-20 text-green-500 animate-pulse" />
              <p className="text-xl text-slate-600 dark:text-slate-300 font-medium">Your Estimated Annual Salary (USD):</p>
              <p className="text-6xl font-extrabold tracking-tighter text-slate-900 dark:text-slate-100 animate-fade-in">
                $ <AnimatedCounter toValue={prediction} />
              </p>
              <CardDescription className="text-gray-500 dark:text-gray-400 text-base italic">
                This is an estimate based on market data and your provided profile.
              </CardDescription>
            </motion.div>
            
            <div className="w-full max-w-4xl grid grid-cols-1 gap-8">
              {featureImportances && (
                <FeatureImportanceChart data={featureImportances} />
              )}

              {salaryDistribution && (
                <SalaryDistributionChart data={salaryDistribution} predictedSalary={prediction} />
              )}
            </div>
          </>
        )}
        <Button 
          onClick={onRestart} 
          className="mt-8 px-8 py-3 rounded-full bg-indigo-600 hover:bg-indigo-700 text-white font-semibold shadow-lg transition-all duration-300 ease-in-out transform hover:scale-105"
        >
          Start Over
        </Button>
      </CardContent>
    </Card>
  );
};