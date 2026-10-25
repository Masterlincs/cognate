declare global {
  namespace App {
    interface PageState {
      e?: string; // the explorer's history entry id (src/lib/history.js)
    }
  }
}

export {};
