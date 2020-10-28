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
