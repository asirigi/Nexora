import { Message } from '@/types';
import { cn } from '@/lib/utils';
import { User, Bot } from 'lucide-react';

interface ChatBubbleProps {
  message: Message;
  className?: string;
}

export const ChatBubble = ({ message, className }: ChatBubbleProps) => {
  const isUser = message.sender === 'user';

  return (
    <div
      className={cn(
        "flex gap-3 p-4 chat-bubble",
        isUser ? "flex-row-reverse" : "flex-row",
        className
      )}
    >
      <div
        className={cn(
          "flex h-8 w-8 shrink-0 select-none items-center justify-center rounded-lg",
          isUser
            ? "bg-chat-user text-chat-user-foreground"
            : "bg-chat-assistant text-chat-assistant-foreground"
        )}
      >
        {isUser ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
      </div>
      
      <div
        className={cn(
          "flex flex-col space-y-2 text-sm max-w-xs lg:max-w-md",
          isUser ? "items-end" : "items-start"
        )}
      >
        <div
          className={cn(
            "px-4 py-2 rounded-2xl shadow-sm",
            isUser
              ? "bg-chat-user text-chat-user-foreground rounded-br-sm"
              : "bg-chat-assistant text-chat-assistant-foreground rounded-bl-sm border border-border"
          )}
        >
          <p className="whitespace-pre-wrap leading-relaxed">{message.content}</p>
        </div>
        
        <span className="text-xs text-muted-foreground px-2">
          {message.timestamp.toLocaleTimeString()}
        </span>
      </div>
    </div>
  );
};