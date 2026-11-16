import Dashboard from "./components/Dashboard";
import EvaluationDashboard from "./components/EvaluationDashboard";

export default function App() {
  if (window.location.pathname === "/evaluation") return <EvaluationDashboard />;
  return <Dashboard />;
}
