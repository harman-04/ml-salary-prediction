import { SalaryPredictor } from './components/SalaryPredictor';

function App() {
  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950 text-slate-800 dark:text-slate-200 flex flex-col items-center justify-center p-4 selection:bg-blue-500/20">
      <div className="text-center mb-10">
        <h1 className="text-4xl sm:text-5xl font-bold tracking-tight bg-gradient-to-r from-blue-600 via-green-500 to-indigo-400 text-transparent bg-clip-text">
          DevSalary AI
        </h1>
        <p className="mt-3 text-lg text-slate-600 dark:text-slate-400 max-w-2xl">
          An interactive journey to estimate your developer salary using a trained LightGBM model.
        </p>
      </div>
      
      <main className="w-full max-w-2xl">
        <SalaryPredictor />
      </main>

      <footer className="mt-10 text-center text-sm text-slate-500">
        <p>Built with React, Flask, and Framer Motion.</p>
      </footer>
    </div>
  );
}

export default App;