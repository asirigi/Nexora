import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { DatabaseVisualization } from '@/components/DatabaseVisualization';
import { Database, Play, Loader2, Table as TableIcon, BarChart3, Settings } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

interface QueryResult {
  columns: string[];
  rows: any[][];
  totalRows: number;
  executionTime: number;
}

interface TableInfo {
  name: string;
  rowCount: number;
  columnCount: number;
  type: 'table' | 'view';
}

export const DatabaseSection = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const [sqlQuery, setSqlQuery] = useState('SELECT * FROM users LIMIT 10;');
  const [queryResult, setQueryResult] = useState<QueryResult | null>(null);
  const [isExecuting, setIsExecuting] = useState(false);
  const [tables] = useState<TableInfo[]>([
    { name: 'users', rowCount: 1247, columnCount: 8, type: 'table' },
    { name: 'orders', rowCount: 5834, columnCount: 12, type: 'table' },
    { name: 'products', rowCount: 892, columnCount: 15, type: 'table' },
    { name: 'customer_analytics', rowCount: 1247, columnCount: 6, type: 'view' },
  ]);
  
  const { toast } = useToast();

  const executeQuery = async () => {
    if (!sqlQuery.trim() || isExecuting) return;

    setIsExecuting(true);
    
    try {
      // Simulate API call to your SQL database
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      // Mock response - replace with actual API call
      const mockResult: QueryResult = {
        columns: ['id', 'name', 'email', 'created_at', 'status'],
        rows: [
          [1, 'John Doe', 'john@example.com', '2024-01-15 10:30:00', 'active'],
          [2, 'Jane Smith', 'jane@example.com', '2024-01-16 14:20:00', 'active'],
          [3, 'Bob Johnson', 'bob@example.com', '2024-01-17 09:45:00', 'inactive'],
          [4, 'Alice Wilson', 'alice@example.com', '2024-01-18 16:15:00', 'active'],
          [5, 'Charlie Brown', 'charlie@example.com', '2024-01-19 11:30:00', 'pending'],
        ],
        totalRows: 1247,
        executionTime: 245,
      };
      
      setQueryResult(mockResult);
      
      toast({
        title: "Query Executed Successfully",
        description: `Returned ${mockResult.rows.length} rows in ${mockResult.executionTime}ms`,
      });
    } catch (error: any) {
      toast({
        title: "Query Error",
        description: error.message || "Failed to execute query",
        variant: "destructive",
      });
    } finally {
      setIsExecuting(false);
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="border-b border-border bg-database-primary/5">
        <div className="p-4">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-8 h-8 rounded-lg bg-database-primary flex items-center justify-center">
              <Database className="h-5 w-5 text-database-primary-foreground" />
            </div>
            <div>
              <h2 className="text-lg font-semibold">Database Analytics</h2>
              <p className="text-sm text-muted-foreground">Connect to your database and explore data insights</p>
            </div>
          </div>

          {/* Database Connection Section */}
          <div className="mb-4 space-y-4">
            <h3 className="text-md font-medium">Data Sources</h3>
            
            <Card className="cursor-pointer hover:bg-secondary/50 transition-colors">
              <CardContent className="p-4">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-secondary flex items-center justify-center">
                    <Database className="h-5 w-5 text-muted-foreground" />
                  </div>
                  <div>
                    <p className="font-medium">Connect to Database</p>
                    <p className="text-sm text-muted-foreground">MySQL, PostgreSQL, MongoDB, and more</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Button variant="secondary" className="w-fit">
              <Settings className="h-4 w-4 mr-2" />
              Configure Connection
            </Button>
          </div>
          
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="overview" className="flex items-center gap-2">
                <Database className="h-4 w-4" />
                Overview
              </TabsTrigger>
              <TabsTrigger value="query" className="flex items-center gap-2">
                <Play className="h-4 w-4" />
                SQL Query
              </TabsTrigger>
              <TabsTrigger value="tables" className="flex items-center gap-2">
                <TableIcon className="h-4 w-4" />
                Tables
              </TabsTrigger>
              <TabsTrigger value="visualizations" className="flex items-center gap-2">
                <BarChart3 className="h-4 w-4" />
                Charts
              </TabsTrigger>
            </TabsList>
          </Tabs>
        </div>
      </div>

      <div className="flex-1 min-h-0">
        <ScrollArea className="h-full">
          <div className="p-4">
            <Tabs value={activeTab} onValueChange={setActiveTab}>
              <TabsContent value="overview" className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm font-medium">Total Tables</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold text-database-primary">
                        {tables.filter(t => t.type === 'table').length}
                      </div>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm font-medium">Total Views</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold text-database-secondary">
                        {tables.filter(t => t.type === 'view').length}
                      </div>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm font-medium">Total Records</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold text-database-chart">
                        {tables.reduce((sum, t) => sum + t.rowCount, 0).toLocaleString()}
                      </div>
                    </CardContent>
                  </Card>
                </div>

                <Card>
                  <CardHeader>
                    <CardTitle>Database Schema Overview</CardTitle>
                    <CardDescription>Quick overview of your database structure</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {tables.map((table) => (
                        <div key={table.name} className="flex items-center justify-between p-3 border rounded-lg">
                          <div className="flex items-center gap-3">
                            <TableIcon className="h-4 w-4 text-database-primary" />
                            <div>
                              <div className="font-medium">{table.name}</div>
                              <div className="text-sm text-muted-foreground">
                                {table.columnCount} columns
                              </div>
                            </div>
                          </div>
                          <div className="text-right">
                            <Badge variant={table.type === 'table' ? 'default' : 'secondary'}>
                              {table.type}
                            </Badge>
                            <div className="text-sm text-muted-foreground mt-1">
                              {table.rowCount.toLocaleString()} rows
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="query" className="space-y-4">
                <Card>
                  <CardHeader>
                    <CardTitle>SQL Query Editor</CardTitle>
                    <CardDescription>Execute SQL queries against your database</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="space-y-2">
                      <label className="text-sm font-medium">SQL Query</label>
                      <Textarea
                        value={sqlQuery}
                        onChange={(e) => setSqlQuery(e.target.value)}
                        placeholder="Enter your SQL query..."
                        className="min-h-[120px] font-mono text-sm"
                      />
                    </div>
                    <Button 
                      onClick={executeQuery}
                      disabled={!sqlQuery.trim() || isExecuting}
                      className="w-full bg-database-primary hover:bg-database-primary/90"
                    >
                      {isExecuting ? (
                        <Loader2 className="h-4 w-4 animate-spin mr-2" />
                      ) : (
                        <Play className="h-4 w-4 mr-2" />
                      )}
                      Execute Query
                    </Button>
                  </CardContent>
                </Card>

                {queryResult && (
                  <Card>
                    <CardHeader>
                      <CardTitle>Query Results</CardTitle>
                      <CardDescription>
                        {queryResult.rows.length} of {queryResult.totalRows} rows • 
                        Executed in {queryResult.executionTime}ms
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="border rounded-md">
                        <Table>
                          <TableHeader>
                            <TableRow>
                              {queryResult.columns.map((column) => (
                                <TableHead key={column} className="font-semibold">
                                  {column}
                                </TableHead>
                              ))}
                            </TableRow>
                          </TableHeader>
                          <TableBody>
                            {queryResult.rows.map((row, index) => (
                              <TableRow key={index}>
                                {row.map((cell, cellIndex) => (
                                  <TableCell key={cellIndex} className="font-mono text-sm">
                                    {cell}
                                  </TableCell>
                                ))}
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </div>
                    </CardContent>
                  </Card>
                )}
              </TabsContent>

              <TabsContent value="tables" className="space-y-4">
                <Card>
                  <CardHeader>
                    <CardTitle>Database Tables</CardTitle>
                    <CardDescription>Manage and explore your database tables</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid gap-4">
                      {tables.map((table) => (
                        <Card key={table.name} className="border-l-4 border-l-database-primary">
                          <CardHeader className="pb-3">
                            <div className="flex items-center justify-between">
                              <div className="flex items-center gap-2">
                                <TableIcon className="h-5 w-5 text-database-primary" />
                                <CardTitle className="text-lg">{table.name}</CardTitle>
                                <Badge variant={table.type === 'table' ? 'default' : 'secondary'}>
                                  {table.type}
                                </Badge>
                              </div>
                              <Button variant="outline" size="sm">
                                <Settings className="h-4 w-4 mr-2" />
                                Manage
                              </Button>
                            </div>
                          </CardHeader>
                          <CardContent>
                            <div className="grid grid-cols-2 gap-4 text-sm">
                              <div>
                                <span className="text-muted-foreground">Columns:</span>
                                <span className="ml-2 font-medium">{table.columnCount}</span>
                              </div>
                              <div>
                                <span className="text-muted-foreground">Rows:</span>
                                <span className="ml-2 font-medium">{table.rowCount.toLocaleString()}</span>
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="visualizations">
                <DatabaseVisualization />
              </TabsContent>
            </Tabs>
          </div>
        </ScrollArea>
      </div>
    </div>
  );
};