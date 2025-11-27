import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { SearchResult } from '@/components/SearchResult';
import { chatAPI } from '@/lib/api';
import { SearchResult as SearchResultType } from '@/types';
import { Search, Loader2, Database } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { ScrollArea } from '@/components/ui/scroll-area';

export const SearchSection = () => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResultType[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);
  const { toast } = useToast();

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || isLoading) return;

    setIsLoading(true);
    setHasSearched(true);

    try {
      const response = await chatAPI.search(query.trim());
      
      if (response.success) {
        // Transform API response to SearchResult format
        const searchResults = Array.isArray(response.data) 
          ? response.data.map((item: any, index: number) => ({
              id: index.toString(),
              title: item.title || item.name || `Result ${index + 1}`,
              content: item.content || item.description || item.text || JSON.stringify(item),
              relevance: item.relevance || item.score || Math.random() * 0.5 + 0.5,
            }))
          : response.data?.results || [{
              id: '0',
              title: 'Search Result',
              content: response.data?.message || JSON.stringify(response.data),
              relevance: 0.9,
            }];
        
        setResults(searchResults);
      } else {
        throw new Error(response.error);
      }
    } catch (error: any) {
      toast({
        title: "Search Error",
        description: error.message || "Failed to perform search",
        variant: "destructive",
      });
      setResults([]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="border-b border-border p-4">
        <form onSubmit={handleSearch} className="flex gap-2">
          <Input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter your search query..."
            disabled={isLoading}
            className="flex-1"
            autoFocus
          />
          <Button 
            type="submit" 
            variant="search" 
            disabled={!query.trim() || isLoading}
          >
            {isLoading ? (
              <Loader2 className="h-4 w-4 animate-spin mr-2" />
            ) : (
              <Search className="h-4 w-4 mr-2" />
            )}
            Search
          </Button>
        </form>
      </div>
      
      <div className="flex-1 min-h-0">
        <ScrollArea className="h-full">
          <div className="p-4">
            {!hasSearched ? (
              <div className="text-center text-muted-foreground py-12">
                <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-search-highlight/10 flex items-center justify-center">
                  <Database className="h-8 w-8 text-search-highlight" />
                </div>
                <h3 className="text-lg font-medium mb-2">Search your data</h3>
                <p className="text-sm max-w-md mx-auto">
                  Enter a query to search through your data and get relevant results with highlighted matches.
                </p>
              </div>
            ) : isLoading ? (
              <div className="space-y-4">
                {[...Array(3)].map((_, i) => (
                  <div key={i} className="bg-search-result border border-border rounded-lg p-4 animate-pulse">
                    <div className="h-4 bg-secondary rounded w-3/4 mb-2"></div>
                    <div className="h-3 bg-secondary rounded w-full mb-1"></div>
                    <div className="h-3 bg-secondary rounded w-2/3"></div>
                  </div>
                ))}
              </div>
            ) : results.length > 0 ? (
              <>
                <div className="mb-4 text-sm text-muted-foreground">
                  Found {results.length} result{results.length !== 1 ? 's' : ''} for "{query}"
                </div>
                <div className="space-y-4">
                  {results.map((result) => (
                    <SearchResult key={result.id} result={result} query={query} />
                  ))}
                </div>
              </>
            ) : (
              <div className="text-center text-muted-foreground py-12">
                <Search className="h-12 w-12 mx-auto mb-4 text-muted-foreground/50" />
                <h3 className="text-lg font-medium mb-2">No results found</h3>
                <p className="text-sm max-w-md mx-auto">
                  No results were found for "{query}". Try adjusting your search terms.
                </p>
              </div>
            )}
          </div>
        </ScrollArea>
      </div>
    </div>
  );
};