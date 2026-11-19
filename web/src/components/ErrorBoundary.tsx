import { Component, type ErrorInfo, type ReactNode } from "react";

interface State {
  failed: boolean;
}

/** Keeps a failure in one panel (e.g. the map) from blanking the whole app. */
export default class ErrorBoundary extends Component<{ children: ReactNode; label: string }, State> {
  state: State = { failed: false };
  static getDerivedStateFromError(): State {
    return { failed: true };
  }
  componentDidCatch(err: Error, info: ErrorInfo) {
    console.error(`${this.props.label} crashed`, err, info.componentStack);
  }
  render() {
    if (!this.state.failed) return this.props.children;
    return (
      <div className="stage-empty" role="alert">
        <p className="stage-empty-title">{this.props.label} failed to render</p>
        <button type="button" className="btn ghost sm" onClick={() => this.setState({ failed: false })}>
          Try again
        </button>
      </div>
    );
  }
}
