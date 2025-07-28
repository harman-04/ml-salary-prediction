// src/components/SalaryPredictor.jsx
import { useState } from 'react';
import { useForm, FormProvider } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';

import { predictionSchema } from '@/lib/schema';
import { Step1 } from './steps/Step1';
import { Step2 } from './steps/Step2';
import { Step3 } from './steps/Step3';
import { ResultDisplay } from './ResultDisplay';
import { Progress } from '@/components/ui/progress';
import { Button } from '@/components/ui/button';
import { jobSatMapping } from '@/lib/data';

const steps = [
  { id: 1, component: Step1 },
  { id: 2, component: Step2 },
  { id: 3, component: Step3 },
];

const variants = {
  enter: (direction) => ({
    x: direction > 0 ? 30 : -30,
    opacity: 0,
  }),
  center: {
    x: 0,
    opacity: 1,
  },
  exit: (direction) => ({
    x: direction < 0 ? 30 : -30,
    opacity: 0,
  }),
};

export const SalaryPredictor = () => {
  const [currentStep, setCurrentStep] = useState(0);
  const [direction, setDirection] = useState(1);
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [featureImportances, setFeatureImportances] = useState(null); // New state for charts
  const [salaryDistribution, setSalaryDistribution] = useState(null); // New state for charts

  const formMethods = useForm({
    resolver: zodResolver(predictionSchema),
    mode: 'onChange',
    defaultValues: {
      YearsCodePro: '',
      WorkExp: '',
      Age: '',
      EdLevel: '',
      DevType: '',
      Country: '',
      OrgSize: '',
      RemoteWork: '',
      Employment: '',
      JobSat: '',

      LanguageHaveWorkedWith: [],
      DatabaseHaveWorkedWith: [],
      PlatformHaveWorkedWith: [],
      WebframeHaveWorkedWith: [],
      MiscTechHaveWorkedWith: [],
      ToolsTechHaveWorkedWith: [],
      NEWCollabToolsHaveWorkedWith: [],
      OfficeStackAsyncHaveWorkedWith: [],
      OfficeStackSyncHaveWorkedWith: [],
      AISearchDevHaveWorkedWith: [],
      AIToolHaveWorkedWith: [],
      Knowledge_1: "",
      Knowledge_2: "",
      Frequency_1: "",
      JobSatPoints_1: "",
       CompTotal: '',
    }
  });

  const { trigger, handleSubmit, reset, formState: { errors } } = formMethods;

  const triggerValidation = async () => {
    switch (currentStep) {
      case 0:
        return await trigger(["YearsCodePro", "WorkExp", "Age", "Country", "OrgSize", "Employment"]);
      case 1:
        return await trigger(["EdLevel", "DevType", "RemoteWork", "JobSat"]);
      case 2: // THIS LINE HAS BEEN UPDATED!
        return await trigger([
          "LanguageHaveWorkedWith", "DatabaseHaveWorkedWith", "PlatformHaveWorkedWith",
          "WebframeHaveWorkedWith", "MiscTechHaveWorkedWith", "ToolsTechHaveWorkedWith",
          "NEWCollabToolsHaveWorkedWith", "OfficeStackAsyncHaveWorkedWith",
          "OfficeStackSyncHaveWorkedWith", "AISearchDevHaveWorkedWith",
          "AIToolHaveWorkedWith",
          "Knowledge_1", "Knowledge_2", "Frequency_1", "JobSatPoints_1" // New fields
        ]);
      default:
        return true;
    }
  };

  const handleNext = async () => {
    const isValid = await triggerValidation();
    if (isValid) {
      setDirection(1);
      setCurrentStep(prev => prev + 1);
    } else {
      console.log("Validation errors:", errors);
    }
  };

  const handleBack = () => {
    setDirection(-1);
    setCurrentStep(prev => prev - 1);
  };

  const onSubmit = async (data) => {
    const finalData = {
      ...data,
      YearsCodePro: data.YearsCodePro === '' ? 0 : data.YearsCodePro,
      WorkExp: data.WorkExp === '' ? 0 : data.WorkExp,
      JobSat: jobSatMapping[data.JobSat] || 0,

       CompTotal: data.CompTotal === '' ? null : data.CompTotal, 
      joinMultiSelect: (arr) => (arr && arr.length > 0 ? arr.join(';') : ''),
    };

    finalData.LanguageHaveWorkedWith = finalData.joinMultiSelect(finalData.LanguageHaveWorkedWith);
    finalData.DatabaseHaveWorkedWith = finalData.joinMultiSelect(finalData.DatabaseHaveWorkedWith);
    finalData.PlatformHaveWorkedWith = finalData.joinMultiSelect(finalData.PlatformHaveWorkedWith);
    finalData.WebframeHaveWorkedWith = finalData.joinMultiSelect(finalData.WebframeHaveWorkedWith);
    finalData.MiscTechHaveWorkedWith = finalData.joinMultiSelect(finalData.MiscTechHaveWorkedWith);
    finalData.ToolsTechHaveWorkedWith = finalData.joinMultiSelect(finalData.ToolsTechHaveWorkedWith);
    finalData.NEWCollabToolsHaveWorkedWith = finalData.joinMultiSelect(finalData.NEWCollabToolsHaveWorkedWith);
    finalData.OfficeStackAsyncHaveWorkedWith = finalData.joinMultiSelect(finalData.OfficeStackAsyncHaveWorkedWith);
    finalData.OfficeStackSyncHaveWorkedWith = finalData.joinMultiSelect(finalData.OfficeStackSyncHaveWorkedWith);
    finalData.AISearchDevHaveWorkedWith = finalData.joinMultiSelect(finalData.AISearchDevHaveWorkedWith);
    finalData.AIToolHaveWorkedWith = finalData.joinMultiSelect(finalData.AIToolHaveWorkedWith);

    // No need to delete joinMultiSelect if it's not a top-level property of the schema
    // delete finalData.joinMultiSelect; 

    setIsLoading(true);
    setError(null);
    setPrediction(null);
    setFeatureImportances(null); // Reset chart data
    setSalaryDistribution(null); // Reset chart data

    try {
      console.log("Sending data to API:", finalData);
      const response = await axios.post('http://localhost:5000/api/predict', finalData);
      
      setPrediction(response.data.predicted_salary_usd);
      setFeatureImportances(response.data.feature_importances || null); // Capture new data
      setSalaryDistribution(response.data.salary_distribution || null); // Capture new data

      setDirection(1);
      setCurrentStep(steps.length);
    } catch (err) {
      console.error("Prediction API Error:", err.response?.data || err.message);
      setError(err.response?.data?.error || `An unexpected error occurred during prediction. Status: ${err.response?.status || 'N/A'}`);
      setDirection(1);
      setCurrentStep(steps.length);
    } finally {
      setIsLoading(false);
    }
  };

  const handleRestart = () => {
    setPrediction(null);
    setError(null);
    setFeatureImportances(null); // Reset on restart
    setSalaryDistribution(null); // Reset on restart
    reset();
    setDirection(-1);
    setCurrentStep(0);
  };

  const CurrentStepComponent = steps[currentStep]?.component;
  const progress = currentStep < steps.length ? ((currentStep + 1) / (steps.length + 1)) * 100 : 100;

  return (
  <FormProvider {...formMethods}>
    <div className="w-full relative h-[650px] sm:h-[500px] md:h-[700px] lg:h-[800px]">
      <div className="mb-4">
        <Progress
          value={progress}
          className="w-full h-2 rounded-full bg-gray-200 dark:bg-gray-700"
        />        </div>

        <AnimatePresence initial={false} custom={direction} mode="wait">
          <motion.div
            key={currentStep}
            custom={direction}
            variants={variants}
            initial="enter"
            animate="center"
            exit="exit"
            transition={{ type: "spring", stiffness: 300, damping: 30, duration: 0.3 }}
            className="absolute w-full"
          >
            {currentStep < steps.length ? (
              <form onSubmit={handleSubmit(onSubmit)}>
                {CurrentStepComponent && <CurrentStepComponent form={formMethods} />}

                <div className="mt-8 flex justify-between items-center">
                  <Button type="button" variant="outline" onClick={handleBack} disabled={currentStep === 0} className="px-6 py-2 rounded-md transition-colors duration-200 hover:bg-gray-100 dark:hover:bg-gray-800">
                    Back
                  </Button>
                  {currentStep < steps.length - 1 ? (
                    <Button type="button" onClick={handleNext} className="px-6 py-2 rounded-md bg-blue-600 hover:bg-blue-700 text-white transition-colors duration-200">
                      Next
                    </Button>
                  ) : (
                    <Button type="submit" disabled={isLoading} className="px-6 py-2 rounded-md bg-green-600 hover:bg-green-700 text-white transition-colors duration-200">
                      {isLoading ? 'Calculating...' : 'Get Prediction'}
                    </Button>
                  )}
                </div>
              </form>
            ) : (
              <ResultDisplay 
                prediction={prediction} 
                error={error} 
                onRestart={handleRestart}
                featureImportances={featureImportances} // Pass new props
                salaryDistribution={salaryDistribution} // Pass new props
              />
            )}
          </motion.div>
        </AnimatePresence>
      </div>
    </FormProvider>
  );
};