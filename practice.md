### Greedy 

<details> 
<summary> <a href="https://leetcode.com/problems/set-intersection-size-at-least-two/"> LC </a> Find min size set which has atleast 2 intersecting elements with each of the given interval </summary>

    int intersectionSizeTwo(vector<vector<int>>& intervals) {
        int n = intervals.size();
        
        sort(intervals.begin(), intervals.end(), [&] (const vector<int>& a, const vector<int>& b) {
            if(a[1] == b[1])
                return a[0] > b[0]; // ** ensures smaller partitions are considered first which are ending at same point
            
            return a[1] < b[1];
        });
        

        int l = intervals[0][1]-1, h = intervals[0][1], cnt=2;
        // l and h are second last and last elements in our req set respec
        for(int i=1; i<n; i++) {            
            int s = intervals[i][0], e = intervals[i][1];
            if(s <= l)
                continue;
            
            ++cnt;
            
            l = h;        
            
            if(s > h) {
                ++cnt;
                l = e-1;
            }
            
            h = e;  // greedily selecting last element of current interval if needed to add atleast 1 more element in this iteration
        }
        
        return cnt;
    }

</details>

<details>
<summary> <a href="https://leetcode.com/problems/find-k-pairs-with-smallest-sums/"> LC </a> Find K pairs with smallest sum in two arrays </summary>

    vector<vector<int>> kSmallestPairs(vector<int>& nums1, vector<int>& nums2, int k) {
        int n1=nums1.size(), n2=nums2.size();
        
        /*** M1 ***/
        if(!n1 || !n2)
            return {};
        
        set<vector<int>> st;
        // set<vector<int>> vis;
        
        st.insert({nums1[0] + nums2[0], 0, 0});
        
        vector<vector<int>> res;
        
        while(k && !st.empty()) {
            
            auto curr = *st.begin();
            st.erase(st.begin());
            k--;
            
            res.push_back({nums1[curr[1]], nums2[curr[2]]});
            
            vector<int> v1({curr[1]+1, curr[2]}), v2({curr[1], curr[2]+1});
            
            if(v1[0] < n1 && v1[1] < n2) {
                // vis.insert(v1);
                st.insert({nums1[v1[0]]+nums2[v1[1]], v1[0], v1[1]});
            }
            
            if(v2[0] < n1 && v2[1] < n2) {
                // vis.insert(v2);
                st.insert({nums1[v2[0]]+nums2[v2[1]], v2[0], v2[1]});
            }
        }
        
        return res;
        

        /***  M2 ***/
        O(k*n1)
        
        
        int next[n1]; // next[i] = next index of nums2 to be paired up with nums1[i]
        memset(next, 0, sizeof next);
        
        vector<vector<int>> res;
        
        while(k > 0) {
            int curr_min = INT_MAX;
            int idx = -1;
            
            for(int i=0; i<n1; i++) {
                if(next[i] < n2 && nums1[i] + nums2[next[i]] < curr_min) {
                    curr_min = nums1[i] + nums2[next[i]];
                    idx = i;
                }
            }
            
            if(idx < 0)
                break;
            
            res.push_back({nums1[idx], nums2[next[idx]]});
            next[idx]++;
            k--;
        }
        
        return res;
            
    }

</details>

### String

<details>
<summary> <a href="https://leetcode.com/problems/shortest-palindrome/"> LC </a> Longest Palindromic prefix <b> KMP </b> </summary>

</details>

### DP

<details>
<summary> <a href="https://codingcompetitions.withgoogle.com/kickstart/round/000000000019ff49/000000000043b0c6"> KS </a> <a href="https://codeforces.com/blog/entry/53960"><b>(Digit DP)</b> </a> Numbers in given range [l, r] having odd digit at odd positions and even at even. pos(msb)=1 </summary>

    ll d[20], dp[20][2];
    int sz;

    ll rec(string& r, int pos, bool less) {
        if(pos >= sz) return 1;
        ll &ans = dp[pos][less];

        if(ans != -1) return ans;
        ll res = 0;

        int hi = 9, c = r[pos]-'0';
        if(!less) hi = c;
        rep(i, 0, hi) if(pos%2 != i%2) res += rec(r, pos+1, less | (i < c));
     
        return ans = res;
    }

    ll calc(string& r) {
        memset(dp, -1, sizeof dp);
        sz = r.size();
        ll res = 0;
        rep(i, 1, sz-1) res += d[i];
        return res + rec(r, 0, 0);

    }

    void go() {
        ll x, y;
        cin>>x>>y;
        string l, r;
        x--;
        l = to_string(x);
        r = to_string(y);



        cout<<calc(r)-calc(l)<<"\n";

    }

    int main(){
        FIO;

        memset(d, 0, sizeof d);
        d[0] = 1;
        for(ll i=1; i<=18; i++) {
            d[i] = 5 * d[i-1];
        }

        int t;
        cin>>t;
        all(t) {
            cout<<"Case #"<<i+1<<": ";
            go();
        }
    }
</details>

<details>
<summary> <a href="https://codeforces.com/contest/505/problem/C"> CF </a> (<b> DP optimisation </b>) Like Frog Jump. jump allowed -> prev-1, prev, prev+1. Collect max gems </summary>

    const int nax=30001;
    ll dp[nax][600], gems[nax];
    int d, OFFSET;
     
    ll rec(int pos, int jmp) {
        // cout<<pos<<" "<<jmp<<"\n";
        if(pos>=nax || jmp<=0) return 0;
        ll &ans = dp[pos][jmp-OFFSET];
        if(ans != -1) return ans;
        return ans = gems[pos] + max({rec(pos+jmp-1, jmp-1), rec(pos+jmp, jmp), rec(pos+jmp+1, jmp+1)});
    }
     
    int main(){
        FIO;
        memset(gems, 0, sizeof gems);
     
        int n,p;
        cin>>n>>d;
        OFFSET = max(0, d-250);
        all(n) cin>>p, gems[p]++;
     
        memset(dp, -1, sizeof dp);
        cout<<rec(d, d)<<"\n";
     
    }

</details>


<details>
<summary> We are given the prices of k products over n days, and we want to buy each product exactly once. However, we are allowed to buy at most one product in a day. What is the minimum total price? </summary>

    void test_case() {
        const int INF=1e6;
        int n, k;   // n-> #days, k-> #products
        cin>>n>>k;
        int prices[k][n];
        int dp[1<<k][n];
        for(int i=0; i<k; i++) {
            for(int j=0; j<n; j++) cin>>prices[i][j];
        }


        memset(dp, 0, sizeof dp);
        for(int i=1; i<(1<<k); i++) {
            for(int j=0; j<n; j++) dp[i][j] = INF;
        }

        for(int x=0; x<k; x++) dp[1<<x][0] = prices[x][0];

        for(int i=0; i<(1<<k); i++) {
            for(int j=1; j<n; j++) {
                dp[i][j] = dp[i][j-1];
                for(int x=0; x<k; x++) {
                    if(i&(1<<x))
                        dp[i][j] = min(dp[i][j], dp[i^(1<<x)][j-1]+prices[x][j]);
                }
            }
        }

        cout<<dp[(1<<k)-1][n-1]<<"\n";
    }

</details>

<details>
<summary> <a href="https://codeforces.com/contest/466/problem/D"> https://codeforces.com/contest/466/problem/D</a>
</summary>

</details>

<details>
<summary> <a href="https://codeforces.com/contest/161/problem/D"><b>DP on trees</b> </a> #pair of nodes with dist = k </summary>

    const int nax=50005;
    vector<int> adj[nax];
    ll dp[nax][505], res;

    // dp[i][j] = #nodes in subtree of i at distance j from i

    int n, k;

    void dfs(int u, int p) {
        for(int v: adj[u]) {
            if(v != p) {
                dfs(v, u);
                for(int i=1; i<=k; i++) {
                    res += dp[u][i] * dp[v][k-i-1];
                }

                for(int i=1; i<=k; i++) {
                    dp[u][i] += dp[v][i-1];
                }
            }
        }

        res += dp[u][k];

    }


    void go() {
        int u, v;
        cin>>n>>k;

        all(n-1) {
            cin>>u>>v;
            --u, --v;
            adj[u].pb(v);
            adj[v].pb(u);
        }

        memset(dp, 0, sizeof dp);
        all(n) dp[i][0] = 1;
        res = 0;

        dfs(0, -1);

        // all(n) {
            // for(int j=0; j<=k; j++)cout<<dp[i][j]<<" ";
            // cout<<'\n';
        // }
        cout<<res<<'\n';

    }

</details>

<details> 
<summary> <a href="https://codeforces.com/problemset/problem/1155/D"> _CF_ </a> Variation of Kadane's </summary>

    void solve() {
        int n, x;
        cin >> n >> x;

        all(n) cin >> a[i];
        ll dp[3] = {};

        ll res=0;
        for(int i=0; i<n; i++) {
            dp[2] = max(dp[2], dp[1]);
            dp[2] = max(0ll, dp[2]+a[i]);

            dp[1] = max({0ll, dp[1]+x*a[i], dp[0] + x*a[i]});

            dp[0] = max(0ll, dp[0]+a[i]);



            res = max({res, dp[0], dp[1], dp[2]});
        }

        cout << res << '\n';
    }

</details>

<details> 
<summary> <a href="https://cses.fi/problemset/task/1665"> CSES </a> Knapsack + open close interval technique <a href="https://usaco.guide/gold/knapsack?lang=cpp"> (See Usaco guide) </a> </summary>

</details>

### Sliding Window

<details>
<summary> Given K sorted lists. Find smallest range covering an element from each list -> min(max(ai - aj)) where i != j </summary>


 <a href="https://leetcode.com/problems/smallest-range-covering-elements-from-k-lists/"> LC </a>

        vector<int> smallestRange(vector<vector<int>>& nums) {
        int k = nums.size();
        
        vector<pair<int, int>> v;
        for(int i=0; i<k; i++) {
            for(int num: nums[i]) v.push_back(make_pair(num, i));
        }
        
        sort(v.begin(), v.end());
        // n distinct elements in a sliding window
        
        int cnt = 0, res = INT_MAX, x=0, y=0;
        unordered_map<int, int> mp;
        for(int i=0, j=0; j<v.size(); j++) {
            int idx = v[j].second, val = v[j].first;
            if(++mp[idx] == 1) cnt++;
            
            while(cnt == k) {
                if(res > val - v[i].first) {res = val - v[i].first; x = v[i].first, y = val;}
                if(--mp[v[i++].second] == 0) cnt--;
            }
        }
        
        return vector<int>({x, y});
    }


<a href="https://codeforces.com/contest/1435/problem/C"> CF (Variation) </a>

        void go() {
        ll a[6]; all(6) cin>>a[i];
        int n; cin>>n;
        ll b[n+5];
        vector<pll> v;
 
        all(n){
            cin>>b[i];
            rep(j, 0, 5) v.pb(make_pair(b[i]-a[j], i));
        }
 
        sort(v.begin(), v.end());
        unordered_map<ll, ll> mp;
        ll res = LLONG_MAX, cnt=0;
 
        for(int i=0, j=0; i<(int)v.size(); i++) {
            ll x = v[i].F, idx = v[i].S;
            if(++mp[idx] == 1) cnt++;
 
            while(cnt >= n) {
                res = min(res, x-v[j].F);
 
                if(--mp[v[j].S] == 0) --cnt;
                j++;
            }
        }
 
        cout<<res<<endl;
 
    } 

</details> 

### Graphs

<details>
<summary> Finding cycle in a <b>grid</b> </summary>

    const int mxN=55;
    string s[mxN];
    bool vis[mxN][mxN];
    int n, m;

    bool dfs(int i, int j, int froi, int froj, char c) {
        // cout<<i<<" "<<j<<endl;
        if(min(i,j)<0 || i>=n || j>=m || s[i][j] != c) return 0;

        if(vis[i][j]) return 1;

        vis[i][j] = 1;

        bool f = 0;
        if(i+1^froi || j^froj)
            f |= dfs(i+1, j, i, j, c);
        if(i-1^froi || j^froj)
            f |= dfs(i-1, j, i, j, c);
        if(i^froi || j+1^froj)
            f |= dfs(i, j+1, i, j, c);
        
        if(i^froi || j-1^froj)
            f |= dfs(i, j-1, i, j, c);

        return f;
    }

    void test_case() {
        cin>>n>>m;

        all(n) cin>>s[i];
        memset(vis, 0, sizeof vis);

        rep(i, 0, n-2) {
            rep(j, 0, m-2) {
                if(!vis[i][j])  {
                    if(dfs(i, j, -1, -1, s[i][j])) {cout<<"Yes\n"; return;}
                }
            }
        }


        cout<<"No\n";

    }

</details>

<details>
<summary> <a href="https://onlinejudge.org/index.php?option=com_onlinejudge&Itemid=8&page=show_problem&problem=3138"> UVA </a> Extra operation in Union-Find => 2 p q Move p to the set containing q </summary>

    /*****
    * set{u} represent set containing u
    * parent array will be 2*n
    * root element of a set will always be [n+1, 2*n] so we don't have to check if curr element is root or not and operatons becomes:
    * union(u, v) ->(Move whole set u) -> connect set{u} and set{v} => par[root(set{u})] = root(set{v})
    * move(u, v) -> (Move only u) -> connect u to set{v} => par[u] = root(set{v})
    *****/

    const int nax=2e5+5;
    int par[nax];
    ll sz[nax], sm[nax];
    int n;


    void init(int n) {
        rep(i, 1, n) {
            par[i] = par[i+n] = i+n;
            sz[i+n] = 1;
            sm[i+n] = i;
        }
    }
    int f(int u) {
        if(u == par[u]) return u;
        return par[u] = f(par[u]);
    }

    void un(int u, int v) {
        u = f(u);
        v = f(v);

        // printf("un %d %d\n", u, v);
        if(u != v) {
            if(sz[v] > sz[u]) swap(u, v);
            par[v] = u;
            sz[u] += sz[v];
            sm[u] += sm[v];
        }
    }

    void move(int u, int v) {
        int pu = f(u);
        v = f(v);

        if(pu != v)  {
            sz[pu]--;
            sm[pu] -= u;

            sz[v]++;
            sm[v] += u;

            par[u] = v;

        }

    }

    void trace() {
        printf("---------\npar: ");
        rep(j, 1, 2*n) printf("%d ", par[j]);
        printf("\n*********\n");
    }

    int main(){
        FIO;

        int t,m,p,q;

        while(cin>>n>>t) {
            init(n);
            all(t) {
                // trace();
                cin>>m>>p;
                if(m == 3) {
                    p = f(p);
                    printf("%lld %lld\n", sz[p], sm[p]);
                }
                else {
                    cin>>q;
                    if(m == 1) un(p, q);
                    else move(p, q);
                }
            }
        }
    }
</details>

<details>
<summary> <a href="https://www.spoj.com/problems/QUEEN/"> SPOJ </a> In a grid min steps to move from F to S. Moves allowed -> chess queen </summary>

    const int nax=1005;
    string gr[nax];
    int n, m;
     
    int dx[]={-1,-1,-1,0,0,1,1,1};
    int dy[]={-1,0,1,-1,1,-1,0,1};
     
    int main(){
        FIO;
     
        int t; cin>>t;
        while(t--) {
            cin>>n>>m;
            all(n+2) gr[i] = string(nax, 'X');
            int si,sj,ti,tj;
            rep(i, 1, n) {
                rep(j, 1, m) {
                    cin>>gr[i][j];
                    if(gr[i][j] == 'F') si=i, sj=j;
                    if(gr[i][j] == 'S') ti=i, tj=j;
                }
            }
     
            int dist[n+1][m+1];
            memset(dist, -1, sizeof dist);
            dist[si][sj] = 0;
     
            queue<pii> qu; 
            qu.push({si, sj});
     
            // bfs -> 1 move away -> 2 moves away -> 3 moves away........
            while(!qu.empty()) {
                int i=qu.front().F, j=qu.front().S;
                qu.pop();
                if(i==ti && j==tj) break;
     
                for(int k=0; k<8; k++) {
                    int x=i+dx[k], y=j+dy[k];
     
                    while(gr[x][y] != 'X') {
                        // At any point of time dist[x][y] can only be <= 1+dist[i][j] if
                        // x,y is already visited
                        if(dist[x][y] == -1) {
                            // case 1: (x,y) -> not visited yet
                            dist[x][y] = 1+dist[i][j];
                            qu.push({x,y});
                        }
                        else if(dist[x][y] < 1+dist[i][j]) break; // case 2: (x,y) is already in queue for lesser dist
                        // we will just continue in case dist[x][y] == 1+dist[i][j]
     
                        x += dx[k], y += dy[k];
                    }
                }
            }
     
            if(dist[ti][tj] >= 0) cout<<dist[ti][tj]<<"\n";
            else cout<<"-1\n";
        }
    }

</details>

### Advanced DS (Range Queries)

<details>
<summary> <a href="https://codeforces.com/contest/1354/problem/D"> _CF_ </a> <a href="geeksforgeeks.org/order-statistic-tree-using-fenwick-tree-bit/"> Order Statistics tree </a> using Fenwick tree </summary>

    // O(n*logn*logn)
    const int mxN = 1e6;
    int bit[mxN+1]; // 1-based indexing

    void update(int idx, int delta) {
        while(idx<=mxN) {
            bit[idx] += delta;
            idx += idx & -idx;
        }
    }

    int sum(int idx) {
        int res = 0;
        while(idx>=1) {
            res += bit[idx];
            idx -= idx & -idx;
        }

        return res;
    }

    void go() {
        int n, q, a, k;
        cin>>n>>q;
        memset(bit, 0, sizeof bit);

        all(n) {
            cin>>a;
            update(a, 1);
        }

        all(q) {
            cin>>k;
            if(k>0){update(k, 1); continue;}
            k *= -1;
            int l=1, h=mxN, mid;
            while(l<h) {
                mid = (l+h) >> 1;
                
                if(sum(mid) >= k) h = mid;
                else l = mid+1;
            }

            update(l, -1);
        }


        all(mxN+1) 
            if(bit[i]>0){cout<<i<<endl; return;}
        cout<<"0\n";
    }

</details>

<details>
<summary> Iterative Segment Tree <b>(Sum): Point update Range Query</b> </summary>


    const int N=2e5+5;    // array size limit
    int n;              // array size
    ll tree[2*N];

    /***
     *
     * segment tree -> 1-indexed and original array -> 0-indexed
     * segment tree :           1.....n, n+1....2*n-1
     * original array elements:       n.........2*n-1
     *
     ***/

    void init(ll *arr) {
        for(int i=n; i<2*n; i++) tree[i] = arr[i-n];
        for(int i=n-1; i>0; i--) tree[i] = tree[i<<1] + tree[i<<1 | 1];


    }

    // idx -> 0..n-1
    // arr[idx] += delta
    void update(int idx, ll delta) {
        idx += n;
        tree[idx] += delta;
        for(; idx>1; idx >>= 1) tree[idx >> 1] = tree[idx] + tree[idx^1];
    }

    // 0<= l <= r < n-1;
    // returns sum [l,r]
    ll query(int l, int r) {
        ll res = 0;
        l += n, r += n;

        for(; l<=r; l=(l+1)>>1, r=(r-1)>>1) {
            if(l&1) res += tree[l];
            if(!(r&1)) res += tree[r];
        }

        return res;
    }


</details>

<details> 
<summary>Iterative Segment Tree <b>RMQ: Point update Range Query</b> </summary>

    const int N=2e5+5;    // array size limit
    int n;              // array size
    ll tree[2*N];
     
    /***
     *
     * segment tree -> 1-indexed and original array -> 0-indexed
     * segment tree :           1.....n, n+1....2*n-1
     * original array elements:       n.........2*n-1
     *
     ***/
     
    void init(ll *arr) {
        for(int i=n; i<2*n; i++) tree[i] = arr[i-n];
        for(int i=n-1; i>0; i--) tree[i] = min(tree[i<<1], tree[i<<1 | 1]);
     
    }
     
    // idx -> 0..n-1
    // arr[idx] = val
    void update(int idx, ll val) {
        idx += n;
        tree[idx] = val;
        for(; idx>1; idx >>= 1) tree[idx >> 1] = min(tree[idx], tree[idx ^ 1]);
    }
     
    // 0<= l <= r < n-1;
    // returns min [l,r]
    ll query(int l, int r) {
        ll res = 1e18;
        l += n, r += n;
     
        for(; l<=r; l=(l+1)>>1, r=(r-1)>>1) {
            res = min({res, tree[l], tree[r]});
        }
     
        return res;
    }

</details>

<details>
<summary> Recursive Segment Tree <b> Lazy Propagation -> Range update range/point query </b> </summary>

    const int N=2e5+5;    // array size limit
    int n;              // array size
    ll arr[N], tree[4*N], lazy[4*N];

    void build(int idx, int lo, int hi) {
        if(lo == hi) {
            tree[idx] = arr[lo];
            return;
        }

        int mid = (lo + hi) >> 1;
        build(idx<<1, lo, mid);
        build(idx<<1 | 1, mid+1, hi);
        tree[idx] = tree[idx<<1] + tree[idx<<1|1];
    }

    void print() {
        all(2*n) cout<<tree[i]<<" ";
        cout<<'\n';
    }

    void init() {
        memset(tree, 0, sizeof tree);
        memset(lazy, 0, sizeof lazy);
        build(1, 0, n-1);

    }

    void pushme(int idx, int lo, int hi) {
        ll& laz = lazy[idx];
        if(laz) {
            tree[idx] += (hi-lo+1) * laz;

            if(lo != hi) {
                lazy[idx<<1] += laz;
                lazy[idx<<1|1] += laz;
            }

            laz = 0;
        }

    }

    void update(int idx, int lo, int hi, int ul, int ur, int delta) {
        if(lo>hi) return;
        pushme(idx, lo, hi);
        if(ur<lo || ul>hi) return;

        if(ul<=lo && hi<=ur) {
            lazy[idx] += delta;
            return;
        }

        int mid = (lo + hi) >> 1;
        update(idx<<1, lo, mid, ul, ur, delta);
        update(idx<<1|1, mid+1, hi, ul, ur, delta);

        tree[idx] = tree[idx<<1] + tree[idx<<1|1];
    }

    ll query(int idx, int lo, int hi, int ql, int qr) {
        if(lo>hi) return 0;
        pushme(idx, lo, hi);
        if(qr<lo || ql>hi) return 0;

        if(ql<=lo && hi<=qr) return tree[idx];
        int mid = (lo+hi) >> 1;
        return query(idx<<1, lo, mid, ql, qr) + query(idx<<1|1, mid+1, hi, ql, qr);
    }

</details>

### Advanced Techniques

<details>
<summary> DSU on Trees / Heavy light decomposition <a href="https://codeforces.com/blog/entry/44351"> blog </a> </summary>


<a href="https://codeforces.com/contest/600/problem/E">ques</a>

<b><a href="https://codeforces.com/contest/600/submission/110491219"> code1 </a> O(Nlog^2N) using map </b>

<b><a href="https://codeforces.com/contest/600/submission/110492873"> code2 </a> O(NlogN) HLD (Heavy light decomp) </b>

*Idea: When merging all subtrees (like merging freq maps in this ques) for any particular vertex, rather than merging
<b>all</b> maps into a new, merge <b>all except with largest size (subtree with max nodes) </b> into that largest
one. So during merging <b> smaller to larger </b> each node will move <b> maximum logn times </b> because size
double on each merge.*

*code2 is an HLD optimization which doesn't uses map*

</details>

<details>
<summary> Divide and conquer and bitwise operations <a href="https://codeforces.com/contest/1416/problem/C"> ques </a> </summary>

</details> 

### General (Mathematics)

<details>
<summary> nCr mod using Ferment little thm  </summary>
    
    // Ferment little thm ->  x/y mod m = x * inv(y) % m
    // inv(y) = pow(y, m-2) mod m
    // M2 -> precomputing inverse factorials 
    // ifac[i] = pow(fac[i], mod-2) = pow(fac[i+1]/(i+1), mod-2) = (i+1) * ifac[i+1]

    const int mxN = 3e5+5, mod=998244353;
    ll fac[mxN], ifac[mxN];

    ll powf(ll a, ll b, ll p) {
        if(!b) return 1;
        ll res = 1;
        while(b) {
            if(b&1) res = res * a % p;
            b = b >> 1;
            a = a*a % mod;
        }

        return res;
    }

    void init(ll n) {
        fac[0] = 1;
        ll i;
        for(i=1; i<=n; i++) fac[i] = i*fac[i-1] % mod;
        i--;

        // M2
        ifac[i] = powf(fac[i], mod-2, mod);
        i--;
        for(; i>=0; i--) ifac[i] = (i+1) * ifac[i+1] % mod;
    }

    ll nCr(ll n, ll r) {
        if(n<r || n<0 || r<0) return 0;
        return (fac[n] * powf(fac[r], mod-2, mod) % mod) *powf(fac[n-r], mod-2, mod) % mod;
        // M2
        // return fac[n] * ifac[r] % mod * ifac[n-r] % mod;
    }

</details>


<details>
<summary> Prime Factorization O(sqrt(n)) </summary>

    unordered_map<ll, ll> primes;
    void primeF(ll n) {
        while(m%2 == 0) ++primes[2], n/=2;

        for(ll i=3; i<=sqrt(n); i+=2) {
            while(n%i == 0) ++primes[i], n/=i;
        }

        if(n>1) primes[n]++;
    }

</details>

<details>
<summary> Prime Factorization using sieve =>  O(nloglogn) precomputation and O(logn) for each query </summary>

    const int mxN=1e5;
    ll spf[mxN];    // spf[i] = smallest prime factor of i
    unordered_map<ll, ll> primes;

    void sieve() {
        iota(spf, spf+mxN, 0);

        for(ll i=2; i*i<mxN; i++) {
            if(spf[i] == i) {
                for(ll j=i*i; j<mxN; j+=i)
                    if(spf[j] == j) spf[j] = i;
            }
        }
    }

    void primeS(ll n) {
        while(n>1) {
            ++primes[spf[n]];
            n /= spf[n];
        }
    }

</details>

<details>
<summary> For efficient string concatenation use += </summary>
</details>

<details> 
<summary> lexicographically <b> kth permutation </b> of a sequence with <b> distinct elements </b> </summary>

<a href=https://codeforces.com/contest/1443/problem/E> Practice Problem (CF) </a>

    const int mxN=15;
    ll fac[mxN+1];
    void fact() {
        fac[0] = 1;
        for(ll i=1; i<=mxN; i++) fac[i] = i*fac[i-1];
    }

    vector<ll> kth_permutaion(vector<ll> seq, ll k) {
        // k >= 0
        // seq -> sorted sequence of elements to be permuted
        // k = 0 -> sorted sequence
        ll n = seq.size();
        vector<ll> res; // kth premutation
        for(ll pos=0; pos<n; pos++) {
            ll idx = k / fac[n-pos-1];
            res.push_back(seq[idx]);

            seq.erase(seq.begin() + idx); // visited array can also be used
            k -= idx * fac[n-pos-1];
        }

        return res;
    }


</details>

<details>
<summary> lexicographically <b> kth permutation </b> of a sequence <b> repetetions allowed </b> </summary>

    const int mxC=26;
    string kth_permutation(string s, int k) {
        int n = s.size();
        int freq[mxC] = {0};
        for(char c: s) freq[c-'a']++;
        string res = "";

        for(int pos=0; pos<n; pos++) {
           for(int c=0; c<mxC; c++) { // try placing char c at index pos?
               if(!freq[c]) continue;

               freq[c]--;
               ll curr = fac[n-pos-1];
               for(int i=0; i<mxC; i++) curr /= fac[freq[i]];

               if(curr > k) {
                   res += (c + 'a');
                   break;
               }

               freq[c]++;
               k -= curr;
           }
        }

        return res;
    }

    // special case -> if sequence have only max 2 distinct kind of characters then nCr can be used
    // calculate curr (See below problem)

<a href="https://leetcode.com/problems/kth-smallest-instructions/submissions/"> LC </a>

</details>

<details>
<summary>  <a href="https://onlinejudge.org/index.php?option=onlinejudge&Itemid=99999999&page=show_problem&category=0&problem=1266&mosmsg=Submission+received+with+ID+25695859"> UVa </a> Find number of integers [1, N] divisible by atleast one of the given M integers <b> Inclusion Exclusion </b> </summary>


<a href="https://cp-algorithms.com/combinatorics/inclusion-exclusion.html"> Explanation </a>

    ll n, m;

    void go() {
        vll a;
        all(m) {
            ll x;
            scanf("%lld", &x);
            if(x == 1) m--;
            else a.pb(x);
        }

        ll res = 0;
        for(int mask=1; mask < (1<<m); mask++) {
            ll lcm = 1, bits=0;
            all(m) {
                if(mask & (1<<i)) {
                    bits++;
                    lcm = lcm*a[i] / __gcd(lcm, a[i]);
                }
            }

            if(bits&1) res += (n/lcm);
            else res -= (n/lcm);
            // printf("%d %lld %lld\n", mask, bits, (n/lcm));
        }

        printf("%lld\n", n-res);
    }

    int main(){
        FIO;

        while(scanf("%lld%lld", &n, &m) == 2) go();
    }

</details>

<details> 
<summary> a+b = (a xor b) + 2*(a and b) </summary>

    Think of a+b = (a XOR b) + (a AND b)*2 as exactly what happen when you do binary addition. From your example, a = 010 and b = 111:

     010
     111
     ---
    1001 = 101 + 100
    For each bit, you add bits from a and b (0+0=0, 0+1=1, 1+0=1, 1+1=0, which is exactly a XOR b plus the carry-in bit from the previous addition, i.e. if both previous bits for a and b are 1, then we add it also. This is exactly (a AND b)*2. (Remember that multiplication by 2 is a shift left.)

    With that equation we can calculate a AND b.

    Now to count the number you want, we look at each bits of a XOR b and a AND b one-by-one and multiply all possibilities. (Let me write a[i] for the i-th bit of a)

    If a[i] XOR b[i] = 0 and a[i] AND b[i] = 0, then a[i] = b[i] = 0. Only one possibility for this bit.

    If a[i] XOR b[i] = 0 and a[i] AND b[i] = 1, then a[i] = b[i] = 1. Only one possibility for this bit.

    If a[i] XOR b[i] = 1 and a[i] AND b[i] = 0, then a[i] = 1 and b[i] = 0 or vice versa. Two possibilities.

    It's not possible to have a[i] XOR b[i] = 1 and a[i] AND b[i] = 1.

    From your example, a XOR b = 101 and a AND b = 010. We have the answer 2*1*2 = 4.
</details>

<details>
<summary> Generating all submasks for masks <b> O(3^n) </b></summary>

    for(int mask=1; mask<1<<n; mask++) {
        // submask = 0 will lead to infinite loop
        for(int submask=mask; submask>0; submask = (submask-1)&mask) {
            // ....
        }
    }

</details>

<details>
<summary> Modular arithmetic </summary>
    A cool property of fractions taken modulo 998244353 (or any other number such that denominator is coprime with it) is that if we want to add two fractions together and calculate the result modulo some number, we can convert these fractions beforehand and then just add them as integer numbers. The same works with subtracting, multiplying, dividing and exponentiating fractions.

</details>
