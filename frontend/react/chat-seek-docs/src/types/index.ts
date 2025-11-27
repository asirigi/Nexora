export interface Message {
  id: string;
  content: string;
  sender: 'user' | 'assistant';
  timestamp: Date;
}

export interface SearchResult {
  id: string;
  title: string;
  content: string;
  relevance?: number;
}

export interface ApiResponse {
  success: boolean;
  message?: string;
  data?: any;
  error?: string;
}