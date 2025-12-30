import os
import json
import re
from datetime import datetime
from typing import List, Dict, Any
from github import Github, GithubException
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

class GitHubClient:
    """Client for extracting data from GitHub repository"""
    
    def __init__(self, token: str, repo_owner: str = None, repo_name: str = None):
        self.token = token
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.client = Github(self.token)
        self.repo = None

        if self.repo_owner and self.repo_name:
            try:
                self.repo = self.client.get_repo(f"{self.repo_owner}/{self.repo_name}")
                print(f"✓ Connected to repository: {self.repo_owner}/{self.repo_name}")
            except GithubException as e:
                print(f"Error connecting to repository: {e}")

    def get_open_issues(self) -> List[Dict[str, Any]]:
        """Fetch all open issues from the repository"""
        if not self.repo:
            return []

        issues = []
        try:
            print("Fetching open issues...")
            for issue in self.repo.get_issues(state='open'):
                if not issue.pull_request:  # Exclude PRs
                    issues.append({
                        'number': issue.number,
                        'title': issue.title,
                        'state': issue.state,
                        'created_at': issue.created_at.isoformat(),
                        'updated_at': issue.updated_at.isoformat(),
                        'labels': [label.name for label in issue.labels],
                        'assignees': [assignee.login for assignee in issue.assignees],
                        'body': issue.body or ""
                    })
            print(f"✓ Fetched {len(issues)} open issues")
        except GithubException as e:
            print(f"Error fetching open issues: {e}")

        return issues

    def get_closed_issues(self, since_days: int = 30) -> List[Dict[str, Any]]:
        """Fetch closed issues from the repository"""
        if not self.repo:
            return []

        issues = []
        try:
            print("Fetching closed issues...")
            for issue in self.repo.get_issues(state='closed'):
                if not issue.pull_request:
                    issues.append({
                        'number': issue.number,
                        'title': issue.title,
                        'state': issue.state,
                        'created_at': issue.created_at.isoformat(),
                        'closed_at': issue.closed_at.isoformat() if issue.closed_at else None,
                        'labels': [label.name for label in issue.labels],
                        'body': issue.body or ""
                    })
            print(f"✓ Fetched {len(issues)} closed issues")
        except GithubException as e:
            print(f"Error fetching closed issues: {e}")

        return issues

    def get_issue_by_id(self, issue_number: int) -> Dict[str, Any]:
        """Get a specific issue by its number"""
        if not self.repo:
            return {}

        try:
            issue = self.repo.get_issue(issue_number)
            return {
                'number': issue.number,
                'title': issue.title,
                'state': issue.state,
                'created_at': issue.created_at.isoformat(),
                'updated_at': issue.updated_at.isoformat(),
                'closed_at': issue.closed_at.isoformat() if issue.closed_at else None,
                'labels': [label.name for label in issue.labels],
                'assignees': [assignee.login for assignee in issue.assignees],
                'body': issue.body or "",
                'comments': issue.comments
            }
        except GithubException as e:
            print(f"Error fetching issue #{issue_number}: {e}")
            return {}

    def get_pull_requests(self, state: str = 'open') -> List[Dict[str, Any]]:
        """Fetch pull requests from the repository"""
        if not self.repo:
            return []

        prs = []
        try:
            print(f"Fetching {state} pull requests...")
            for pr in self.repo.get_pulls(state=state):
                prs.append({
                    'number': pr.number,
                    'title': pr.title,
                    'state': pr.state,
                    'created_at': pr.created_at.isoformat(),
                    'updated_at': pr.updated_at.isoformat(),
                    'merged_at': pr.merged_at.isoformat() if pr.merged_at else None,
                    'author': pr.user.login,
                    'labels': [label.name for label in pr.labels],
                    'base': pr.base.ref,
                    'head': pr.head.ref,
                    'body': pr.body or ""
                })
            print(f"✓ Fetched {len(prs)} {state} pull requests")
        except GithubException as e:
            print(f"Error fetching pull requests: {e}")

        return prs

    def get_active_contributors(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get list of active contributors"""
        if not self.repo:
            return []

        contributors = []
        try:
            print("Fetching contributors...")
            for contributor in self.repo.get_contributors()[:limit]:
                contributors.append({
                    'login': contributor.login,
                    'contributions': contributor.contributions,
                    'avatar_url': contributor.avatar_url,
                    'type': contributor.type
                })
            print(f"✓ Fetched {len(contributors)} contributors")
        except GithubException as e:
            print(f"Error fetching contributors: {e}")

        return contributors

    def extract_last_commit(self) -> Dict[str, Any]:
        """Get the most recent commit information"""
        if not self.repo:
            return {}

        try:
            commits = self.repo.get_commits()
            latest_commit = commits[0]
            return {
                'sha': latest_commit.sha,
                'message': latest_commit.commit.message,
                'author': latest_commit.commit.author.name,
                'date': latest_commit.commit.author.date.isoformat(),
                'url': latest_commit.html_url
            }
        except GithubException as e:
            print(f"Error fetching last commit: {e}")
            return {}

    def get_commit_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent commit history"""
        if not self.repo:
            return []

        commits = []
        try:
            print(f"Fetching commit history (limit: {limit})...")
            for commit in self.repo.get_commits()[:limit]:
                commits.append({
                    'sha': commit.sha,
                    'message': commit.commit.message,
                    'author': commit.commit.author.name,
                    'date': commit.commit.author.date.isoformat(),
                    'url': commit.html_url
                })
            print(f"✓ Fetched {len(commits)} commits")
        except GithubException as e:
            print(f"Error fetching commit history: {e}")

        return commits

    def get_branches(self) -> List[Dict[str, Any]]:
        """Get all branches in the repository"""
        if not self.repo:
            return []

        branches = []
        try:
            print("Fetching branches...")
            for branch in self.repo.get_branches():
                branches.append({
                    'name': branch.name,
                    'protected': branch.protected,
                    'commit_sha': branch.commit.sha
                })
            print(f"✓ Fetched {len(branches)} branches")
        except GithubException as e:
            print(f"Error fetching branches: {e}")

        return branches

    def get_releases(self) -> List[Dict[str, Any]]:
        """Get all releases/tags"""
        if not self.repo:
            return []

        releases = []
        try:
            print("Fetching releases...")
            for release in self.repo.get_releases():
                releases.append({
                    'tag_name': release.tag_name,
                    'name': release.title,
                    'created_at': release.created_at.isoformat(),
                    'published_at': release.published_at.isoformat() if release.published_at else None,
                    'author': release.author.login if release.author else None,
                    'body': release.body or ""
                })
            print(f"✓ Fetched {len(releases)} releases")
        except GithubException as e:
            print(f"Error fetching releases: {e}")

        return releases

    def summarize_repo_state(self) -> Dict[str, Any]:
        """Get a comprehensive summary of the repository state"""
        if not self.repo:
            return {}

        try:
            print("Fetching repository summary...")
            summary = {
                'name': self.repo.name,
                'description': self.repo.description,
                'stars': self.repo.stargazers_count,
                'forks': self.repo.forks_count,
                'open_issues_count': self.repo.open_issues_count,
                'language': self.repo.language,
                'created_at': self.repo.created_at.isoformat(),
                'updated_at': self.repo.updated_at.isoformat(),
                'default_branch': self.repo.default_branch,
                'last_commit': self.extract_last_commit(),
                'active_contributors_count': len(self.get_active_contributors(limit=10))
            }
            print("✓ Repository summary fetched")
            return summary
        except GithubException as e:
            print(f"Error fetching repo state: {e}")
            return {}

    def fetch_all_data(self) -> Dict[str, Any]:
        """Fetch all GitHub data at once"""
        print("\n" + "="*80)
        print("FETCHING GITHUB REPOSITORY DATA")
        print("="*80 + "\n")
        
        data = {
            'repo_summary': self.summarize_repo_state(),
            'open_issues': self.get_open_issues(),
            'closed_issues': self.get_closed_issues(),
            'open_prs': self.get_pull_requests(state='open'),
            'closed_prs': self.get_pull_requests(state='closed'),
            'contributors': self.get_active_contributors(),
            'commits': self.get_commit_history(),
            'branches': self.get_branches(),
            'releases': self.get_releases(),
            'last_updated': datetime.now().isoformat()
        }
        
        print("\n" + "="*80)
        print("DATA EXTRACTION COMPLETED")
        print("="*80 + "\n")
        
        return data


class GitHubDataOrganizer:
    """Organizes GitHub data into structured text chunks for embedding"""
    
    @staticmethod
    def organize_repo_summary(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Organize repository summary into chunks"""
        chunks = []
        
        if summary:
            text = f"""Repository: {summary.get('name', 'Unknown')}
Description: {summary.get('description', 'No description')}
Language: {summary.get('language', 'Unknown')}
Stars: {summary.get('stars', 0)}
Forks: {summary.get('forks', 0)}
Open Issues: {summary.get('open_issues_count', 0)}
Created: {summary.get('created_at', 'Unknown')}
Last Updated: {summary.get('updated_at', 'Unknown')}
Default Branch: {summary.get('default_branch', 'Unknown')}"""
            
            if summary.get('last_commit'):
                lc = summary['last_commit']
                text += f"\nLast Commit: {lc.get('message', '')} by {lc.get('author', '')} on {lc.get('date', '')}"
            
            chunks.append({
                'type': 'repository_summary',
                'text': text,
                'metadata': {'name': summary.get('name', 'Unknown')}
            })
        
        return chunks
    
    @staticmethod
    def organize_issues(issues: List[Dict[str, Any]], issue_type: str) -> List[Dict[str, Any]]:
        """Organize issues into chunks"""
        chunks = []
        
        for issue in issues:
            text = f"""Issue #{issue['number']}: {issue['title']}
State: {issue_type}
Created: {issue['created_at']}
Labels: {', '.join(issue['labels']) if issue['labels'] else 'None'}
Assignees: {', '.join(issue['assignees']) if issue.get('assignees') else 'None'}

Description:
{issue['body'][:500]}"""
            
            chunks.append({
                'type': f'{issue_type}_issue',
                'text': text,
                'metadata': {
                    'number': issue['number'],
                    'title': issue['title'],
                    'labels': issue['labels']
                }
            })
        
        return chunks
    
    @staticmethod
    def organize_pull_requests(prs: List[Dict[str, Any]], pr_state: str) -> List[Dict[str, Any]]:
        """Organize pull requests into chunks"""
        chunks = []
        
        for pr in prs:
            text = f"""Pull Request #{pr['number']}: {pr['title']}
State: {pr_state}
Author: {pr['author']}
Created: {pr['created_at']}
Base: {pr['base']} ← Head: {pr['head']}
Labels: {', '.join(pr['labels']) if pr['labels'] else 'None'}
Merged: {pr.get('merged_at', 'Not merged')}

Description:
{pr['body'][:500]}"""
            
            chunks.append({
                'type': f'{pr_state}_pull_request',
                'text': text,
                'metadata': {
                    'number': pr['number'],
                    'title': pr['title'],
                    'author': pr['author']
                }
            })
        
        return chunks
    
    @staticmethod
    def organize_commits(commits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Organize commits into chunks"""
        chunks = []
        
        for commit in commits:
            text = f"""Commit: {commit['sha'][:8]}
Author: {commit['author']}
Date: {commit['date']}
Message: {commit['message']}
URL: {commit['url']}"""
            
            chunks.append({
                'type': 'commit',
                'text': text,
                'metadata': {
                    'sha': commit['sha'],
                    'author': commit['author']
                }
            })
        
        return chunks
    
    @staticmethod
    def organize_contributors(contributors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Organize contributors into chunks"""
        chunks = []
        
        if contributors:
            # Create a summary chunk
            text = "Repository Contributors:\n"
            for contributor in contributors:
                text += f"- {contributor['login']}: {contributor['contributions']} contributions\n"
            
            chunks.append({
                'type': 'contributors_summary',
                'text': text,
                'metadata': {'count': len(contributors)}
            })
        
        return chunks
    
    @staticmethod
    def organize_branches(branches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Organize branches into chunks"""
        chunks = []
        
        if branches:
            text = "Repository Branches:\n"
            for branch in branches:
                protected = "Protected" if branch['protected'] else "Not protected"
                text += f"- {branch['name']}: {protected}\n"
            
            chunks.append({
                'type': 'branches_summary',
                'text': text,
                'metadata': {'count': len(branches)}
            })
        
        return chunks
    
    @staticmethod
    def organize_releases(releases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Organize releases into chunks"""
        chunks = []
        
        for release in releases:
            text = f"""Release: {release['tag_name']}
Name: {release.get('name', 'Unnamed')}
Created: {release['created_at']}
Published: {release.get('published_at', 'Not published')}
Author: {release.get('author', 'Unknown')}

Release Notes:
{release['body'][:500]}"""
            
            chunks.append({
                'type': 'release',
                'text': text,
                'metadata': {
                    'tag': release['tag_name'],
                    'name': release.get('name', '')
                }
            })
        
        return chunks
    
    @staticmethod
    def organize_all_data(github_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Organize all GitHub data into structured chunks"""
        print("\n" + "="*80)
        print("ORGANIZING DATA INTO CHUNKS")
        print("="*80 + "\n")
        
        all_chunks = []
        
        # Organize repository summary
        summary_chunks = GitHubDataOrganizer.organize_repo_summary(github_data.get('repo_summary', {}))
        all_chunks.extend(summary_chunks)
        print(f"✓ Organized repository summary: {len(summary_chunks)} chunks")
        
        # Organize open issues
        open_issue_chunks = GitHubDataOrganizer.organize_issues(github_data.get('open_issues', []), 'open')
        all_chunks.extend(open_issue_chunks)
        print(f"✓ Organized open issues: {len(open_issue_chunks)} chunks")
        
        # Organize closed issues
        closed_issue_chunks = GitHubDataOrganizer.organize_issues(github_data.get('closed_issues', []), 'closed')
        all_chunks.extend(closed_issue_chunks)
        print(f"✓ Organized closed issues: {len(closed_issue_chunks)} chunks")
        
        # Organize open PRs
        open_pr_chunks = GitHubDataOrganizer.organize_pull_requests(github_data.get('open_prs', []), 'open')
        all_chunks.extend(open_pr_chunks)
        print(f"✓ Organized open PRs: {len(open_pr_chunks)} chunks")
        
        # Organize closed PRs
        closed_pr_chunks = GitHubDataOrganizer.organize_pull_requests(github_data.get('closed_prs', []), 'closed')
        all_chunks.extend(closed_pr_chunks)
        print(f"✓ Organized closed PRs: {len(closed_pr_chunks)} chunks")
        
        # Organize commits
        commit_chunks = GitHubDataOrganizer.organize_commits(github_data.get('commits', []))
        all_chunks.extend(commit_chunks)
        print(f"✓ Organized commits: {len(commit_chunks)} chunks")
        
        # Organize contributors
        contributor_chunks = GitHubDataOrganizer.organize_contributors(github_data.get('contributors', []))
        all_chunks.extend(contributor_chunks)
        print(f"✓ Organized contributors: {len(contributor_chunks)} chunks")
        
        # Organize branches
        branch_chunks = GitHubDataOrganizer.organize_branches(github_data.get('branches', []))
        all_chunks.extend(branch_chunks)
        print(f"✓ Organized branches: {len(branch_chunks)} chunks")
        
        # Organize releases
        release_chunks = GitHubDataOrganizer.organize_releases(github_data.get('releases', []))
        all_chunks.extend(release_chunks)
        print(f"✓ Organized releases: {len(release_chunks)} chunks")
        
        print(f"\n✓ Total chunks created: {len(all_chunks)}")
        print("="*80 + "\n")
        
        return all_chunks


class GitHubVectorStore:
    """Create embeddings and store GitHub data in FAISS vector database"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        print(f"Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Store metadata
        self.metadata = []
        
        print(f"✓ Model loaded (embedding dimension: {self.embedding_dim})")
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for text chunks"""
        print(f"\nCreating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print("✓ Embeddings created")
        return embeddings
    
    def add_to_index(self, chunks: List[Dict[str, Any]]):
        """Add chunks to FAISS index"""
        if not chunks:
            print("No chunks to add")
            return
        
        print("\n" + "="*80)
        print("CREATING EMBEDDINGS AND STORING IN FAISS")
        print("="*80 + "\n")
        
        # Extract text from chunks
        texts = [chunk['text'] for chunk in chunks]
        
        # Create embeddings
        embeddings = self.create_embeddings(texts)
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store metadata
        self.metadata.extend(chunks)
        
        print(f"\n✓ Added {len(chunks)} chunks to FAISS index")
        print(f"✓ Total vectors in index: {self.index.ntotal}")
        print("="*80 + "\n")
    
    def search(self, query: str, k: int = 5, filter_type: str = None) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using a query, with added heuristics for specific query types.
        """
        if self.index.ntotal == 0:
            print("Index is empty")
            return []

        # --- Heuristic 1: Aggregate/Summary Queries ---
        summary_keywords = ['total', 'count', 'how many', 'summarize', 'summary', 'number of']
        if any(keyword in query.lower() for keyword in summary_keywords):
            summary_chunks = [m for m in self.metadata if 'summary' in m.get('type', '')]
            if summary_chunks:
                print(f"✓ Prioritizing summary chunks for aggregate query: '{query}'")
                # Return all summary chunks, as they are all relevant for aggregates
                return summary_chunks

        # --- Heuristic 2: Direct ID Lookup (e.g., "issue #123", "PR 45") ---
        id_match = re.search(r'(issue|pr|pull request)\s*#?(\d+)', query, re.IGNORECASE)
        if id_match:
            item_type, item_id = id_match.groups()
            item_id = int(item_id)
            
            # Normalize item_type
            if 'pr' in item_type or 'pull' in item_type:
                target_types = ['open_pull_request', 'closed_pull_request']
            else:
                target_types = ['open_issue', 'closed_issue']

            for chunk in self.metadata:
                if chunk.get('type') in target_types and chunk.get('metadata', {}).get('number') == item_id:
                    print(f"✓ Found direct match for {item_type} #{item_id}")
                    return [chunk]

        # --- Heuristic 3: Implicit Filtering (e.g., "latest commits", "open PRs") ---
        query_lower = query.lower()
        potential_filters = []
        if 'commit' in query_lower: potential_filters.append('commit')
        if 'issue' in query_lower: potential_filters.extend(['open_issue', 'closed_issue'])
        if 'pr' in query_lower or 'pull request' in query_lower: potential_filters.extend(['open_pull_request', 'closed_pull_request'])
        if 'branch' in query_lower: potential_filters.append('branches_summary')
        if 'release' in query_lower or 'tag' in query_lower: potential_filters.append('release')
        if 'contributor' in query_lower: potential_filters.append('contributors_summary')

        # --- Fallback to Vector Search ---
        print(f"Performing vector search for query: '{query}'")
        query_embedding = self.model.encode([query])
        
        # Search more results to allow for filtering
        search_k = min(k * 5, self.index.ntotal)
        distances, indices = self.index.search(query_embedding.astype('float32'), search_k)
        
        results = []
        seen_indices = set()
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx in seen_indices:
                continue
            
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                
                # Apply implicit filters if any were found
                if potential_filters and result.get('type') not in potential_filters:
                    continue # Skip this result if it doesn't match the implicit type

                # Apply explicit filter from arguments
                if filter_type and result.get('type') != filter_type:
                    continue

                result['distance'] = float(distances[0][i])
                result['similarity_score'] = 1 / (1 + float(distances[0][i]))
                
                results.append(result)
                seen_indices.add(idx)
                if len(results) >= k:
                    break
        
        # --- Heuristic 4: Recency Sort ---
        if 'latest' in query_lower or 'recent' in query_lower:
            # Sort by date if available in metadata
            results.sort(key=lambda r: r.get('metadata', {}).get('created_at', '1970-01-01'), reverse=True)
            print("✓ Sorted results by recency.")

        return results
    
    def save_index(self, index_path: str = 'github_faiss_index.bin', metadata_path: str = 'github_metadata.pkl', config_path: str = 'github_config.json'):
        """Save FAISS index and metadata to disk"""
        faiss.write_index(self.index, index_path)
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        print(f"✓ Index saved to {index_path}")
        print(f"✓ Metadata saved to {metadata_path}")
    
    def load_index(self, index_path: str = 'github_faiss_index.bin', metadata_path: str = 'github_metadata.pkl'):
        """Load FAISS index and metadata from disk"""
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            print("Index or metadata file not found")
            return
        
        self.index = faiss.read_index(index_path)
        
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        print(f"✓ Loaded index with {self.index.ntotal} vectors")
        print(f"✓ Loaded {len(self.metadata)} metadata entries")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored data"""
        type_counts = {}
        for item in self.metadata:
            item_type = item.get('type', 'unknown')
            type_counts[item_type] = type_counts.get(item_type, 0) + 1
        
        return {
            'total_chunks': len(self.metadata),
            'total_vectors': self.index.ntotal,
            'embedding_dimension': self.embedding_dim,
            'type_distribution': type_counts
        }


# Main execution
if __name__ == "__main__":
    # Configuration
    GITHUB_TOKEN = "YOUR_GITHUB_TOKEN_HERE"
    REPO_OWNER = ""  # Replace with repository owner
    REPO_NAME = ""  # Replace with repository name
    
    # Check if we need to re-fetch data
    config_path = 'github_config.json'
    should_fetch = True
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            # Check if repository is the same
            if (config.get('repo_owner') == REPO_OWNER and 
                config.get('repo_name') == REPO_NAME and
                config.get('token') == GITHUB_TOKEN):
                should_fetch = False
                print("\n✓ Same repository detected. Using existing database.")
                print(f"  Repository: {REPO_OWNER}/{REPO_NAME}")
                print("  To force refresh, delete 'github_config.json' or change repository settings.\n")
    
    if should_fetch:
        print("\n🔄 Fetching fresh data from GitHub repository...")
        
        # Step 1: Extract GitHub data
        github_client = GitHubClient(
            token=GITHUB_TOKEN,
            repo_owner=REPO_OWNER,
            repo_name=REPO_NAME
        )
        
        github_data = github_client.fetch_all_data()
        
        # Save raw data to JSON
        with open('github_raw_data.json', 'w') as f:
            json.dump(github_data, f, indent=2)
        print("✓ Raw data saved to github_raw_data.json\n")
        
        # Step 2: Organize data into chunks
        organized_chunks = GitHubDataOrganizer.organize_all_data(github_data)
        
        # Step 3: Create embeddings and store in FAISS
        vector_store = GitHubVectorStore(model_name='all-MiniLM-L6-v2')
        vector_store.add_to_index(organized_chunks)
        
        # Step 4: Save the vector database
        vector_store.save_index('github_faiss_index.bin', 'github_metadata.pkl')
        
        # Step 5: Save configuration
        config = {
            'repo_owner': REPO_OWNER,
            'repo_name': REPO_NAME,
            'token': GITHUB_TOKEN,
            'last_updated': datetime.now().isoformat()
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✓ Configuration saved to {config_path}\n")
        
        # Step 6: Display statistics
        stats = vector_store.get_statistics()
        print("\n" + "="*80)
        print("VECTOR STORE STATISTICS")
        print("="*80)
        print(f"Total chunks: {stats['total_chunks']}")
        print(f"Total vectors: {stats['total_vectors']}")
        print(f"Embedding dimension: {stats['embedding_dimension']}")
        print("\nData type distribution:")
        for data_type, count in stats['type_distribution'].items():
            print(f"  - {data_type}: {count}")
        print("="*80)
        
        print("\n✅ GitHub data extraction and vectorization completed!")
    else:
        print("✅ Using existing GitHub database. Run with run_ingestion=True to refresh.")
