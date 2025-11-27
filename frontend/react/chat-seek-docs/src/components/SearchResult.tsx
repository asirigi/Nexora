import { SearchResult as SearchResultType } from '@/types';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Search } from 'lucide-react';

interface SearchResultProps {
  result: SearchResultType;
  query: string;
}

export const SearchResult = ({ result, query }: SearchResultProps) => {
  const highlightText = (text: string, query: string) => {
    if (!query) return text;
    
    const regex = new RegExp(`(${query})`, 'gi');
    const parts = text.split(regex);
    
    return parts.map((part, index) => 
      regex.test(part) ? (
        <span key={index} className="bg-search-highlight/20 text-search-highlight font-medium">
          {part}
        </span>
      ) : (
        part
      )
    );
  };

  return (
    <Card className="bg-search-result border-border hover:border-search-highlight/50 transition-colors duration-200">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium text-search-result-foreground flex items-center gap-2">
          <Search className="h-4 w-4 text-search-highlight" />
          {highlightText(result.title, query)}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <p className="text-sm text-muted-foreground leading-relaxed">
          {highlightText(result.content, query)}
        </p>
        {result.relevance && (
          <div className="mt-2 flex items-center gap-2">
            <div className="flex-1 bg-secondary rounded-full h-1">
              <div 
                className="bg-search-highlight h-1 rounded-full transition-all duration-500"
                style={{ width: `${result.relevance * 100}%` }}
              />
            </div>
            <span className="text-xs text-muted-foreground">
              {Math.round(result.relevance * 100)}% match
            </span>
          </div>
        )}
      </CardContent>
    </Card>
  );
};