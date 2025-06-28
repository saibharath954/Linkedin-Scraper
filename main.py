# main.py
import asyncio
import json
import sys
import argparse
from backend.app.scraper.linkedin import LinkedInScraper

async def main():
    parser = argparse.ArgumentParser(description='LinkedIn Company Posts Scraper')
    parser.add_argument('--url', type=str, help='LinkedIn company URL')
    parser.add_argument('--posts', type=int, default=5, help='Number of posts to scrape (default: 5)')
    parser.add_argument('--output', type=str, help='Output file path (optional)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Default URL if none provided
    company_url = args.url or "https://www.linkedin.com/company/microsoft"
    max_posts = args.posts
    
    print(f"Starting LinkedIn scraper...")
    print(f"Company URL: {company_url}")
    print(f"Max posts: {max_posts}")
    print("-" * 50)
    
    # Initialize scraper
    scraper = LinkedInScraper()
    
    # Enable debug mode if requested
    if args.debug:
        scraper.headless = False
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Scrape posts
        posts = await scraper.scrape_company_posts(company_url, max_posts)
        
        if posts:
            print(f"\n✅ Successfully scraped {len(posts)} posts!")
            print("=" * 60)
            
            # Display posts
            for i, post in enumerate(posts, 1):
                print(f"\n📝 POST {i}:")
                print("-" * 30)
                
                # Content
                content = post.get('content', 'No content')
                if len(content) > 200:
                    content = content[:200] + "..."
                print(f"Content: {content}")
                
                # Timestamp
                if post.get('timestamp'):
                    print(f"Timestamp: {post['timestamp']}")
                
                # Engagement
                reactions = post.get('reactions', 0)
                comments = post.get('comments', 0)
                if reactions > 0 or comments > 0:
                    print(f"Engagement: {reactions} reactions, {comments} comments")
                
                # Images
                if post.get('images'):
                    print(f"Images: {len(post['images'])} found")
                
                # Links
                if post.get('links'):
                    print(f"Links: {len(post['links'])} found")
                
                # Post URL
                if post.get('url'):
                    print(f"Post URL: {post['url']}")
            
            # Save to file if requested
            if args.output:
                try:
                    with open(args.output, 'w', encoding='utf-8') as f:
                        json.dump(posts, f, indent=2, ensure_ascii=False)
                    print(f"\n💾 Posts saved to: {args.output}")
                except Exception as e:
                    print(f"\n❌ Error saving to file: {e}")
            
            # Summary
            print(f"\n📊 SUMMARY:")
            print(f"   • Total posts scraped: {len(posts)}")
            print(f"   • Posts with content: {sum(1 for p in posts if p.get('content', '').strip())}")
            print(f"   • Posts with timestamps: {sum(1 for p in posts if p.get('timestamp'))}")
            print(f"   • Posts with engagement: {sum(1 for p in posts if p.get('reactions', 0) > 0 or p.get('comments', 0) > 0)}")
            print(f"   • Posts with images: {sum(1 for p in posts if p.get('images'))}")
            print(f"   • Posts with links: {sum(1 for p in posts if p.get('links'))}")
            
        else:
            print("❌ No posts found or scraping failed")
            print("\n🔧 Troubleshooting tips:")
            print("   • Check if the company URL is correct")
            print("   • Try running with --debug flag to see browser actions")
            print("   • LinkedIn may be blocking requests - try again later")
            print("   • Some companies may have restricted access to their posts")
            
    except KeyboardInterrupt:
        print("\n⚠️ Scraping interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()

def run_interactive():
    """Interactive mode for easier usage"""
    print("🔗 LinkedIn Company Posts Scraper")
    print("=" * 40)
    
    # Get company URL
    company_url = input("Enter LinkedIn company URL (or press Enter for Microsoft): ").strip()
    if not company_url:
        company_url = "https://www.linkedin.com/company/microsoft"
    
    # Get number of posts
    try:
        max_posts = int(input("Number of posts to scrape (default 5): ") or "5")
    except ValueError:
        max_posts = 5
    
    # Debug mode
    debug_mode = input("Enable debug mode? (y/N): ").lower().startswith('y')
    
    # Output file
    output_file = input("Save to file? (Enter filename or press Enter to skip): ").strip()
    
    # Build arguments
    sys.argv = ['main.py', '--url', company_url, '--posts', str(max_posts)]
    if debug_mode:
        sys.argv.append('--debug')
    if output_file:
        sys.argv.extend(['--output', output_file])
    
    # Run main
    asyncio.run(main())

if __name__ == "__main__":
    try:
        # Check if running with arguments
        if len(sys.argv) > 1:
            asyncio.run(main())
        else:
            # Run in interactive mode
            run_interactive()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)