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

### Advanced DS

<details>
<summary> <a href="https://codeforces.com/contest/1354/problem/D"> _CF_ </a> <a href="geeksforgeeks.org/order-statistic-tree-using-fenwick-tree-bit/"> Order Statistics tree </a> using Fenwick tree </summary>

    // O(n*logn*logn)
    const int mxN = 1e6;
    int bit[mxN+1];

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


### General

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

    unordered_map<ll, l> primes;
    void primeF(ll n) {
        while(l%2 == 0) ++primes[2], l/=2;

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

</details>
