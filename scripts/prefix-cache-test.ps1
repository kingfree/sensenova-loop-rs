param([string]$Port = "8091", [int]$MaxTok = 6)
$long = "Here is a short story about a lighthouse keeper named Elias who lived on a rocky island for thirty years. Every evening he climbed the spiral stairs, trimmed the wick, and watched the sea. One stormy night a small boat appeared in the waves, and he rowed out to save the two children inside. They stayed with him until spring, and the island was never quiet again."
function Turn($msgs, $label) {
  $b = @{ model='sensenova-loop-9b'; max_tokens=$MaxTok; messages=$msgs } | ConvertTo-Json -Depth 6
  $t = Measure-Command { $r = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/chat/completions" -Method Post -ContentType 'application/json' -Body $b -TimeoutSec 1200 }
  $c = $r.choices[0].message.content
  Write-Host ("{0} ({1:N2}s): [{2}]" -f $label, $t.TotalSeconds, $c)
  return $c
}
$msgs = @(@{role='user'; content=$long + "`n`nIn one sentence, what did Elias do?"})
$r1 = Turn $msgs "TURN1"
$msgs = $msgs + @(@{role='assistant'; content=$r1}) + @(@{role='user'; content='And where did the children stay?'})
$r2 = Turn $msgs "TURN2"
$msgs = $msgs + @(@{role='assistant'; content=$r2}) + @(@{role='user'; content='What was the keeper name again?'})
$r3 = Turn $msgs "TURN3"

