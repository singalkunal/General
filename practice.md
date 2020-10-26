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

        
